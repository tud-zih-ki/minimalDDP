#########################
#
# main.py
# script trains and tests STEADY meshgraphnets
#
#########################

import random
import slurmTorch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

SEED = 1337
# all experiments with seed=0
torch.manual_seed(SEED)
random.seed(SEED)


def main(args):
    rank = slurmTorch.rank
    torch.set_printoptions(edgeitems=6,sci_mode=True,precision=12)

    
    if slurmTorch.rank == 0:
        print(f'>>> Training on {len(slurmTorch.hostnames)} node(s) and {slurmTorch.size} processes. Master node of hosts {slurmTorch.hostnames} is {slurmTorch.hostnames[0]}')
    print("- Process {} corresponds to GPU {} of node {}".format(slurmTorch.rank, slurmTorch.local_rank, slurmTorch.local_nodeID))

    # Get Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Set up architecture
    model = modelclass(...)
    ddp_model = DDP(model,device_ids=[slurmTorch.local_rank])
    
    # Optimizer
    optim = torch.optim.Adam(ddp_model.parameters(), lr=lr)
    
    if rank == 0:    
        gnn_model_summary(model)
        logging.info('Will use device: {}'.format(device))
        
    dist.barrier() #sync
    ddp_model.train()
    end_now = False
        
    for epoch in range(STARTEPOCH,NEPOCHS):
        for next_data in trainloader:
            step += 1

            x, target, ... = batch_dict_to_device(next_data)

            optim.zero_grad(set_to_none=True)
        
            loss = ddp_model(x, target, ...) 
            
            
            loss.backward()  # DDP syncs and Backward pass.
            optim.step()
            
            # LRscheduler.step()
            
    return None

def gnn_model_summary(model,logger=logging,RESTORED=False):

    model_params_list = list(model.named_parameters())
    logger.info("-"*(40+20+15+10+10))
    line_new = "{:>40}  {:>20}  {:>15} {:<10} {:<10}".format("Layer.Parameter", "Param Tensor Shape", "Param #", "Trainable", "CUDA")
    logger.info(line_new)
    logger.info("-"*(40+20+15+10+10))
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = int(torch.tensor(elem[1].size()).prod().item())
        trainable = elem[1].requires_grad
        cuda      = elem[1].is_cuda
        line_new = "{:>40}  {:>20} {:>15} {:<10} {:<10}".format(p_name, str(p_shape), str(p_count),str(trainable),str(cuda))
        logger.info(line_new)
    logger.info("-"*(40+20+15+10+10))
    total_params = sum([param.nelement() for param in model.parameters()])
    logger.info("Total params:{}".format(total_params))
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: {}".format(num_trainable_params))
    logger.info("Non-trainable params: {}".format(total_params - num_trainable_params))

    if RESTORED:
        logger.info('## NORMALIZER ##')
        for elem in model_params_list:
            p_name = elem[0]
            if 'normalizer' in p_name:
                p_shape = list(elem[1].size())
                p_count = int(torch.tensor(elem[1].size()).prod().item())
                trainable = elem[1].requires_grad
                cuda      = elem[1].is_cuda
                line_new = "{:>40}  {:>20} {:>15} {:<10} {:<10}".format(p_name, str(p_shape), str(p_count),str(trainable),str(cuda))
                logger.info(line_new)
                print(elem[1])

################### MAIN #######################
if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    parser = argparser()

    args = parser.parse_args()
    
    torch.cuda.set_device(slurmTorch.local_rank)
    mp.set_start_method('spawn')

    # USES ENVs: MASTER_ADDR and MASTER_PORT that are set in slurmTorch.py
    dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=slurmTorch.size,
                        rank=slurmTorch.rank)
    
    if slurmTorch.rank == 0:
        logging.info(f'\n\n### Train Distributed Meshgraphnets ###\n Now: {datetime.now().strftime("%Y%m%d-%H%M%S")}')
        logging.info('File: ' + sys.argv[0])       
 
    dist.barrier()
    main(args)
    
    if slurmTorch.rank == 0:
        logging.info('###### TRAINING ENDED ON {} ######\n\n'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))
    
    dist.barrier()
    dist.destroy_process_group()  
    sys.exit(0)
