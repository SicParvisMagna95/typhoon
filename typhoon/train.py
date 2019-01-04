from typhoon.model import *
from typhoon.dataset import *

def train(net,loss_func,optimizer,epochs=EPOCH):
    for epoch in range(epochs):
        print(f'===== Epoch {epoch} =====')

        """训练单个epoch"""
        for batch_idx,(b_x,b_y) in enumerate(train_loader):
            print(f'batch:{batch_idx}')
            if USE_GPU:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            output = net(b_x)
            optimizer.zero_grad()
            loss = loss_func(output, b_y)
            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()

            # '''每个batch输出在测试集上的效果'''
            test_input = location_test_data.cuda()
            # tensor float32 (8231,1,3)
            out_test = net(test_input)
            loss_test = loss_func(out_test.detach().cpu(),
                                  location_test_label)

            # 反归一化得到真实值
            # ndarray float32 (8271,1)
            lat_out = out_test.detach().cpu().numpy()[:,:,0]\
                      *(location_train_data_norm[1]-location_train_data_norm[2])\
                             +location_train_data_norm[2]
            lon_out = out_test.detach().cpu().numpy()[:,:,1]\
                      *(location_train_data_norm[3]-location_train_data_norm[4])\
                             +location_train_data_norm[4]
            # pressure_out = out_test.detach().cpu().numpy()[:,:,2]\
            #                *(location_train_data_norm[5]-location_train_data_norm[6])\
            #                  +location_train_data_norm[6]


            # pressure_errors = np.zeros(lat_out.shape[0])

            distance_time = []
            for j in range(4):
                # 初始化结果 (8271,)
                distance_errors = np.zeros(lat_out.shape[0])
                for i in range(lat_out.shape[0]):
                    distance_errors[i] = distance(lat_out[i,j],lon_out[i,j],
                                                  loc_test_label[i,j,0],loc_test_label[i,j,1])
                    # pressure_errors[i] = abs(pressure_out[i,0] - loc_test_label[i,0,2])
                distance_error = np.mean(distance_errors)
                distance_time.append(distance_error)
            # pressure_error = np.mean(pressure_errors)

            print(f'train loss:{loss.item()}', f'||test loss:{loss_test.item()}',
                  f'||test error:{distance_time}km')
    torch.save(net, 'model.pkl')