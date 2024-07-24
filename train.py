class Train_model():
    def __init__(self,data_handler,window_size,step,object_name,model,optimizer,scheduler,criterion,epoch=500,model_name="MGGTSP-CAT"):
        self.data_handler = data_handler
        self.mode = model
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        eval_metrics = self.train_model(data_handler, window_size, step, object_name, model,criterion, optimizer, scheduler, epochs=epoch,model_name=model_name)

    #---------------------------------------------------#
    #   train model
    #---------------------------------------------------#  
    def train_model(self,data_handler,window_size,step,object_name,model,criterion,optimizer,scheduler,epochs,model_name="MGGTSP-CAT",start_epoch=0,train_size=0.8,validation_size=0.0):

        data = data_handler.data_wt
        train_loader1,test_loader1 = data_handler.train_loader1,data_handler.test_loader1
        train_loader2,test_loader2 = data_handler.train_loader2,data_handler.test_loader2
        train_loader3,test_loader3 = data_handler.train_loader3,data_handler.test_loader3
        
        scaler_y = MinMaxScaler()
        y_label_index = [data.columns.get_loc(index) for index in object_name]
        y_true = data.iloc[:,y_label_index] 
        scaler_y.fit(y_true)
        y_range , y_min = scaler_y.data_range_ , scaler_y.data_min_ 
        y_true = y_true.iloc[window_size:] 
        y_true_test = y_true[int(data_handler.X1.shape[0]*(train_size)):]    
        pd.DataFrame(y_true_test).to_excel(f"y_true_test_{step}.xlsx",index=False) 

        eval_metrics = []
        for epoch in range(epochs):    
    #************************************train set************************************
            model.train()  
            running_loss = 0.0
            for (bx1,by1),(bx2,by2),(bx3,by3) in zip(train_loader1, train_loader2, train_loader3):
                # Forward pass: Compute predicted y by passing x to the model
                y_train_pred = model(bx1, bx2, bx3)  
                # Compute loss
                loss = criterion(y_train_pred, by1)
                running_loss += loss.item()
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
                scheduler.step()
            print(f'Epoch {start_epoch+epoch+1}, Train Loss: {running_loss/len(train_loader1)}')
            
    #************************************test set************************************
            if (epoch+1) % 500 == 0:
                # test set
                y_test_true_list,y_test_pred_list = [],[]
                model.eval()
                with torch.no_grad():
                    for (bx1,by1),(bx2,by3),(bx3,by3) in zip(test_loader1, test_loader2, test_loader3):
                        y_test_pred = model(bx1, bx2, bx3)
                        y_test_pred = y_test_pred.cpu() * y_range + y_min # Denormalization
                        by1 = by1.cpu() *  y_range + y_min 
                        y_test_pred_list.append(y_test_pred)
                        y_test_true_list.append(by1)
                y_test_pred = torch.cat(y_test_pred_list, dim=0).cpu()
                y_test_true = torch.cat(y_test_true_list, dim=0).cpu()
                loss = criterion(y_test_pred, y_test_true)
                print('Epoch: {}, Test loss: {:.5f}'.format(start_epoch+epoch + 1, loss))
                eval_metrics = self.test_eval(window_size,step,object_name,
                                        start_epoch+epoch+1,loss,
                                        y_true_test,y_test_pred,y_test_true,eval_metrics)
                self.show_result(window_size,step,object_name,y_true_test,y_test_pred,y_test_true)
                for i in range(len(object_name)):
                    pd.DataFrame(y_test_pred[:, :, i].detach().numpy()).to_excel(f"{model_name}_y_test_pred_object{i},step{step}.xlsx",index=False) 
                    pd.DataFrame(y_test_true[:, :, i].detach().numpy()).to_excel(f"y_test_true_object{i},step{step}.xlsx",index=False) 

        return eval_metrics
    
    
    def show_result(self,window_size,step,object_name,y_true_test,y_test_pred,y_test_true):

        plt.figure(figsize=(40, 25)) if y_true_test.shape[1] > 1 else plt.figure(figsize=(28, 10))
        for i in range(y_true_test.shape[1]):
            plt.subplot(y_true_test.shape[1], 1, i + 1)
            for j in range(step):
                y_test_pred_i= y_test_pred[:,j][:,i].detach().numpy() 
                y_test_true_i = y_test_true[:,j][:,i].detach().numpy()
                plt.plot(range(j,len(y_test_pred_i)+j), y_test_pred_i,  label='Predict Value:'+str(j+1)+'step',linewidth=0.6)
                plt.title("window_sizes="+str(window_size)+"：predict"+object_name[i]+"in test set", fontsize=15)
            plt.plot(range(0,y_true_test.shape[0]), y_true_test.iloc[:,i], label = 'True Value',linewidth=0.8,linestyle='--',color='red',alpha=0.9) 
            plt.legend(fontsize=12, bbox_to_anchor=(1, 1))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

        plt.show()
        plt.tight_layout()  

 
    def show_eval(self,y_true,y_pred):
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        MAE = mean_absolute_error(y_true,y_pred)
        r2 = r2_score(y_true, y_pred)
        print('RMSE:'+str(RMSE))
        print('MAE:'+str(MAE))
        print('R²:'+str(r2))
        return RMSE,MAE,r2


 
    def test_eval(self,window_size,step,object_name,epoch,total_loss,y_true_test,y_test_pred,y_test_true,eval_metrics):
        eval_string = "Epoch: {}, Test Loss: {:.5f}\n".format(epoch, total_loss.item())
        for i in range(y_true_test.shape[1]):
            for j in range(step):
                y_test_pred_i= y_test_pred[:,j][:,i].detach().numpy() 
                y_test_true_i = y_test_true[:,j][:,i].detach().numpy()
                print("-----"+"epoch="+str(epoch)+"，predict"+object_name[i]+"in test set: step"+str(j+1)+"-----")
                RMSE, MAE, r2 = self.show_eval(y_test_true_i,y_test_pred_i) 
                eval_string += f"predict{object_name[i]}: step {j+1}\n" \
                               f"rmse: {float(RMSE):.5f}\n" \
                               f"mae: {float(MAE):.5f}\n" \
                               f"r²: {float(r2):.5f}\n"
            print("-----"+"epoch="+str(epoch)+"，predict"+object_name[i]+"in test set: total-----")
            all_RMSE, all_MAE, all_r2 = self.show_eval(y_test_true[:,:,i],y_test_pred[:,:,i])
            eval_string += f"predict {object_name[i]}: total \n" \
                               f"rmse: {float(all_RMSE):.5f}\n" \
                               f"mae: {float(all_MAE):.5f}\n" \
                               f"r²: {float(all_r2):.5f}\n"
            print("-----------------------------")
        if(step>1):
            print("-----"+"epoch="+str(epoch)+"，Overall prediction accuracy -----")
            y_test_flat = y_test_true.numpy().reshape(-1, y_test_true.shape[-1])
            y_pred_flat = y_test_pred.numpy().reshape(-1, y_test_pred.shape[-1])
            all_RMSE, all_MAE, all_r2 = self.show_eval(y_test_flat,y_pred_flat)
            eval_string += f"Overall prediction accuracy:\n" \
                               f"rmse: {float(all_RMSE):.5f}\n" \
                               f"mae: {float(all_MAE):.5f}\n" \
                               f"r²: {float(all_r2):.5f}\n"
        eval_metrics.append(eval_string)

        return eval_metrics


    