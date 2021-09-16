import torch
import torch.nn as nn
import copy

from models import my_resnet18

class Flerd(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, model, K=2048, m=0.999, T=0.07, sim_mat=None, dataset='cifar10'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Flerd, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.sim_mat = sim_mat

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = model
        self.encoder_k = copy.deepcopy(model)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # get output dim of the encoder model
        if dataset in ['cifar10', 'cifar100']:
            x = torch.randn([2,3,32,32])
        elif dataset in ['imagenet', 'imagenet100']:
            x = torch.randn([2,3,224,224])
        elif dataset in ['tiny-imagenet']:
            x = torch.randn([2,3,64,64])
        dim = self.encoder_k(x).shape[-1]

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_idx", torch.zeros(K, dtype=torch.long))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, idxs):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_idx[ptr:ptr+batch_size] = idxs
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, batch_idx):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        # with torch.no_grad():
        #     print(q.square().sum(dim=1).mean().detach())
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # dequeue and enqueue
        # dequeue and enqueue first, and this distillation reduces to a soft-version of MoCo training.
        self._dequeue_and_enqueue(k, batch_idx)
            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) / self.T

        # labels: positive key indicators
        labels = self.sim_mat[batch_idx][:, self.queue_idx]     #labels[i,j] = self.sim_mat[batch_idx[i], self.queue_idx[j]]
        # renormalize the labels to make it a probability distribution.
        labels = labels / labels.sum(axis=1, keepdims=True)
       
        return logits, labels, q, k


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    cnn_model = my_resnet18()

    model = Flerd(cnn_model, K=2048, m=0.999, T=0.07, sim_mat=torch.exp(torch.randn((10000, 10000))))

    for i in range(1):
        x = torch.randn([128, 3, 32, 32])
        batch_idxs = torch.randint(low=0, high=10000, size=(128,))
    
    
        res = model(x, x, batch_idxs)

    print(model.encoder_q.state_dict())
