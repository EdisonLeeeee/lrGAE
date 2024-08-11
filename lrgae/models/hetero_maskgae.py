class HeteroMaskGAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        negative_sampler = 'random',
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.loss_fn = HeteroFusedBCE(decoder)
        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler        

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

