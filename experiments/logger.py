from torch.utils.tensorboard import SummaryWriter


class FlowSeqLogger(SummaryWriter):
    def __init__(self, log_dir, ):
        super(FlowSeqLogger, self).__init__(log_dir)

    def log_training(self, recon_loss, kl_loss, llen, kl_mul_weight, loss, learning_rate, kl_weight, batch_size, step):
        self.add_scalar('train/recon_loss', recon_loss, step)
        self.add_scalar('train/kl_loss', kl_loss, step)
        self.add_scalar('train/learning_rate', learning_rate, step)
        self.add_scalar('train/kl_mul_weight', kl_mul_weight, step)
        self.add_scalar('train/llen', llen, step)
        self.add_scalar('train/all_loss', loss, step)
        self.add_scalar('train/kl_weight', kl_weight, step)
        self.add_scalar('train/batch_size', batch_size, step)

    def log_pretrain(self, recon_loss, llen, loss, learning_rate, step):
        self.add_scalar('pre-train/recon_loss', recon_loss, step)
        # self.add_scalar('pre-train/kl_loss', kl_loss, step)
        self.add_scalar('pre-train/learning_rate', learning_rate, step)
        # self.add_scalar('train/kl_mul_weight', kl_mul_weight, step)
        self.add_scalar('pre-train/llen', llen, step)
        self.add_scalar('pre-train/all_loss', loss, step)

    def log_eval(self, recon_loss, kl_loss, llen, epoch):
        self.add_scalar('eval/recon_loss', recon_loss, epoch)
        self.add_scalar('eval/kl_loss', kl_loss, epoch)
        self.add_scalar('eval/llen', llen, epoch)