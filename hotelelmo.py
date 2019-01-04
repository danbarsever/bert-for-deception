import numpy as np
import pickle
import os
import torch
from numpy import array
from numpy import asarray
from numpy import zeros
#import pandas
from collections import Counter
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
#from keras.preprocessing import sequence
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)

np.random.seed(7)

max_rev_length=500
em_vec_len=300
top_words=5000


hs=256

with open('deceptive_elmo_tokens.pkl','rb') as f:
	deceptives=pickle.load(f)
with open('truthful_elmo_tokens.pkl','rb') as f:
	truthfuls=pickle.load(f)
X_train_t=deceptives[0:400]+truthfuls[0:400]
X_test_t=deceptives[400:800]+truthfuls[400:800]
y_train=torch.load('torch_y_train.pt')
y_test=torch.load('torch_y_test.pt')






class Net(nn.Module):
	def __init__(self, batch_size, hidden_size):
		super(Net, self).__init__()
			#self.attn=nn.Linear(50,50)
				#self.attn_com=nn.Linear(50,50)
		output_size=2
		self.hidden_size=hs
		self.batch_size = batch_size

		self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

		self.lstm1=nn.LSTM(input_size=1024, hidden_size=hs,bidirectional=True, dropout=.8, batch_first=True)
		self.W_s1 = nn.Linear(2*hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.fc_layer = nn.Linear(30*2*hidden_size, 1000)
		self.label = nn.Linear(1000, output_size)
		self.out=nn.Softmax(dim=0)
	def attention_net(self, lstm_output):



		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		#print(attn_weight_matrix)
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix
	def forward(self,input, batch_size=None):
		#attn_w=F.softmax(self.attn(torch.cat(input[0],hidden[0]),1),dim=1)
				#attn_a=torch.bmm(attn_w,input)
		e=self.elmo(input)
		e=e['elmo_representations'][0]
		#print(e.shape)
		e=torch.sum(e,dim=0)
		e=e.unsqueeze(0)
		#print(e.shape)

		#print(e.shape)
		if batch_size is None:
			h_0 = Variable(torch.zeros(2, e.shape[0], self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2,  e.shape[0], self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2,  e.shape[0], self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2,  e.shape[0], self.hidden_size).cuda())
		output, (h_n, c_n) = self.lstm1(e, (h_0, c_0))
		#print(e)
		attn_weight_matrix = self.attention_net(output)

		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		#print(input.shape[0])
		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		logits = self.label(fc_out)
		#print(x)
		x=self.out(logits)
		# print(x)
		return x

	def initHidden(self):
		return torch.zeros(1,self.hidden_size)
bs=1
model=Net(weights=attweights, batch_size=bs, hidden_size=hs).cuda()
loss_function=nn.CrossEntropyLoss().cuda()
optimizer=optim.Adagrad(model.parameters())
inds=np.arange(800)
from random import shuffle
shuffle(inds)

for epoch in range(10):
	print('Epoch: '+str(epoch+1))
	for j in range(0,800, bs):
		print('   step: '+str(j+1))
		model.zero_grad()

		inp=X_train_t[inds[j]].cuda()

		y_prd=model(inp, bs)


		tar=y_train[inds[j]].cuda()

		loss=loss_function(y_prd,tar.unsqueeze(0))
		loss.backward()
		optimizer.step()
		
print('made it this far')
testmark=100
firstpart=np.arange(0,0+testmark/2)
secpart=np.arange(400,400+testmark/2)
tp=np.concatenate((firstpart,secpart))
tp=tp.astype('int32')
xt=[X_test_t[x] for x in tp]
yt=[y_test[x] for x in tp]
yt=torch.tensor(yt)
xt=torch.tensor(xt)
scores=model(xt.view(testmark,500).cuda(),testmark)
#scores=model(X_test_t.cuda(),800)
res=torch.ones(testmark).long()
for i in range(testmark):
	res[i]=torch.argmax(scores[i])
print(scores)
print(res)
acc=yt==res
#print(model(X_train_t[200:200+bs].view(500,bs,50).cuda()))
print(torch.sum(acc).numpy()/float(testmark))

