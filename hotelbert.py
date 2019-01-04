import torch
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
#from keras.preprocessing import sequence
from torch.autograd import Variable


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


x_train=torch.load('padded_x_train.pt')
x_test=torch.load('padded_x_test.pt')
x_train_mask=torch.load('train_mask.pt')
x_test_mask=torch.load('test_mask.pt')

y_train=torch.load('torch_y_train.pt')
y_test=torch.load('torch_y_test.pt')

class Net(nn.Module):
	def __init__(self, batch_size):
		super(Net, self).__init__()
		self.hidden_size=256
		self.batch_size=batch_size
		hidden_size=768
		output_size=2
		self.bert = BertModel.from_pretrained('bert-base-uncased')

		self.lstm1=nn.LSTM(input_size=768, hidden_size=256,bidirectional=True, dropout=.5, batch_first=True)
		self.W_s1 = nn.Linear(512, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.fc_layer = nn.Linear(10*2*hidden_size, 1000)
		self.label = nn.Linear(1000, output_size)
		self.out=nn.Softmax(dim=1)
	def attention_net(self, lstm_output):



		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		#print(attn_weight_matrix)
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=1)
		return attn_weight_matrix
	def forward(self, input, mask, batch_size):
		_,x=self.bert(input, attention_mask=mask)
		
		x=x.unsqueeze(1)

		if batch_size is None:
			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
		output, (h_n, c_n) = self.lstm1(x, (h_0, c_0))
		
		#print(e)
		attn_weight_matrix = self.attention_net(output)
		#print(attn_weight_matrix.shape)
		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		#print(input.shape[0])
		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		x=self.label(fc_out)

		return x
bs=5
model=Net(batch_size=bs).cuda()

loss_function=nn.CrossEntropyLoss().cuda()
optimizer=optim.Adagrad(model.parameters(), lr=.5)

inds=np.arange(800)
from random import shuffle



for epoch in range(10):
	shuffle(inds)
	print('Epoch: '+str(epoch+1))
	for j in range(0,800, bs):
		if j % 20 == 0:
			print('   step: '+str(j+1))
		model.zero_grad()

		inp=x_train[inds[j:j+bs]].view(bs,500).cuda()
		mask=x_train_mask[inds[j:j+bs]].view(bs,500).cuda()

		y_prd=model(inp,mask,bs)

		tar=y_train[inds[j:j+bs]].cuda()#.unsqueeze(0)
	
		loss=loss_function(y_prd,tar)
		
		loss.backward()
		optimizer.step()

print('testing time with 100')

testmark=10
firstpart=np.arange(0,0+testmark/2)
secpart=np.arange(400,400+testmark/2)
tp=np.concatenate((firstpart,secpart))
tp=tp.astype('int32')

scores=model(x_test[tp].view(testmark,500).cuda(),x_test_mask[tp].view(testmark,500).cuda(),testmark)
res=torch.ones(testmark).long()
for i in range(testmark):
	res[i]=torch.argmax(scores[i])

corrects=res==y_train[tp]

acc=torch.sum(corrects).numpy()/float(testmark)

print(corrects)
print(acc)
