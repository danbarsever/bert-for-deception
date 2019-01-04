import torch
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
#from keras.preprocessing import sequence
from torch.autograd import Variable
import argparse

from tensorboardX import SummaryWriter
import datetime,socket,os

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

comment='_sequenceclass'
log_dir = os.path.join('runs/'+ 'bert_' + current_time + '_' + socket.gethostname() + comment)
print(log_dir)
writer = SummaryWriter(log_dir = log_dir)

parser = argparse.ArgumentParser()

	## Required parameters
# parser.add_argument("--data_dir",
# 					default=None,
# 					type=str,
# 					required=True,
# 					help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
# parser.add_argument("--bert_model", default=None, type=str, required=True,
# 					help="Bert pre-trained model selected in the list: bert-base-uncased, "
# 						 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
# parser.add_argument("--task_name",
# 					default=None,
# 					type=str,
# 					required=True,
# 					help="The name of the task to train.")
# parser.add_argument("--output_dir",
# 					default=None,
# 					type=str,
# 					required=True,
# 					help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--max_seq_length",
					default=128,
					type=int,
					help="The maximum total input sequence length after WordPiece tokenization. \n"
						 "Sequences longer than this will be truncated, and sequences shorter \n"
						 "than this will be padded.")
parser.add_argument("--do_train",
					default=False,
					action='store_true',
					help="Whether to run training.")
parser.add_argument("--do_eval",
					default=False,
					action='store_true',
					help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case",
					default=False,
					action='store_true',
					help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
					default=32,
					type=int,
					help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
					default=8,
					type=int,
					help="Total batch size for eval.")
parser.add_argument("--learning_rate",
					default=5e-1,
					type=float,
					help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
					default=3.0,
					type=float,
					help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
					default=0.1,
					type=float,
					help="Proportion of training to perform linear learning rate warmup for. "
						 "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
					default=False,
					action='store_true',
					help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
					type=int,
					default=-1,
					help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
					type=int,
					default=42,
					help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
					type=int,
					default=1,
					help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
					default=False,
					action='store_true',
					help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
					type=float, default=0,
					help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
						 "0 (default value): dynamic loss scaling.\n"
						 "Positive power of 2: static loss scaling value.\n")

args = parser.parse_args()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x

x_train=torch.load('padded_x_train.pt')
x_test=torch.load('padded_x_test.pt')
x_train_mask=torch.load('train_mask.pt')
x_test_mask=torch.load('test_mask.pt')

y_train=torch.load('torch_y_train.pt')
y_test=torch.load('torch_y_test.pt')

t_total=800

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout=nn.Dropout(.2)
		self.classifier=nn.Linear(768,2)
		self.out=nn.Softmax(dim=1)
		#self.apply(self.bert.init_bert_weights)
	def forward(self, input):
		_,x=self.bert(input)
		x=self.dropout(x)
		x=self.classifier(x)
		
		x=self.out(x)
		return x

model=BertForSequenceClassification.from_pretrained('bert-base-uncased')
device=torch.device("cuda")
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=args.learning_rate,
							 warmup=args.warmup_proportion,
							 t_total=t_total)
loss_function=nn.CrossEntropyLoss().cuda()
#optimizer=optim.Adagrad(model.parameters(), lr=.5)

inds=np.arange(800)
from random import shuffle

global_step=0
bs=1
model.train()
for epoch in range(200):
	shuffle(inds)
	print('Epoch: '+str(epoch+1))
	tr_loss = 0
	for j in range(0,800, bs):
		if j % 20 == 0:
			print('   step: '+str(j+1))
		#model.zero_grad()

		inp=x_train[inds[j]].cuda()
		mask=x_train_mask[inds[j]].cuda()
		#print(inds[j])
		#inp=inp.reshape([1,500,50])
		tar=y_train[inds[j]].unsqueeze(0).cuda()
		loss=model(inp,attention_mask=mask,labels=tar)
		loss.backward()

		tr_loss += loss.item()	

		lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr_this_step
		optimizer.step()
		optimizer.zero_grad()
		global_step += 1
		#writer.add_scalar('loss', loss.item(), epoch)
		# loss.backward()
		# optimizer.step()
	print(tr_loss)
	writer.add_scalar('bert_'+'/train/'+'loss', tr_loss, epoch)


print('testing time')
res=[]
for i in range(len(x_test)):
	y_prd=model(x_test[i].cuda())
	_, indmax=torch.max(y_prd[0],0)
	res.append(indmax)
res=np.array(res)
res=torch.tensor(res)

corrects=res==y_train

acc=torch.sum(corrects).numpy()/800.

print(corrects)
print(acc)
