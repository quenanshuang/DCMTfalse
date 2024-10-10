import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import ISICDataset
from models import deeplabv3
from utils.utils import dice_score_batch
from sklearn.metrics import roc_auc_score,accuracy_score
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT', help='project path for saving results')
    # parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/pretaincheckpoint/UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='FR_UNet1', choices=['DeepLabv3p', 'UNet','UNet_dsc'], help='segmentation backbone')
    parser.add_argument('--backbonet', type=str, default='FR_UNet1', choices=['DeepLabv3p', 'UNet','UNet_dsc','FR_UNet1'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='/home/quenanshuang/unmt_test/DATA/DRIVE', help='path to the data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.2, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=8, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='lastt.pth', help='model weights')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    test_set = ISICDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False)
    # test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False, pin_memory=True)
    # print("len(test_set)",len(test_set))
    # print(len(test_set))
    return test_dataloder, len(test_set)


def load_model(model_weights, in_channels, num_classes, backbone):
    model = deeplabv3.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model


def eval(is_debug=False):
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(10)
        length = 10 * args.batch_size
    else:
        pbar = range(iters)
    # Load model
    # weights_path = project_path + 'weights_STARE_3frunet_withmse/' + args.model_weights
    weights_path = project_path + 'weights_3frunet_onlyuncertaintyMSE_drive_0829/epoch_19/' + args.model_weights
    model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbonet)
    # model2 = load_model(model_weights="/home/quenanshuang/unmt_test/runs/UCMT_FR_UNet1_label_0.2/weights_3frunet_all_share/last1.pth", in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbonet)
    # model3 = load_model(model_weights="/home/quenanshuang/unmt_test/runs/UCMT_UNet_label_0.2/weights_3unet_all/mselast2.pth", in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbonet)
    model.eval()
    # model2.eval()
    # model3.eval()
    ############################
    # Evaluation
    ############################
    print('start evaluation')
    results = {i: [] for i in range(args.num_classes)}
    score_AUC=[]
    score_ACC=[]
    score_SP=[]
    score_SE=[]
    score_IOU=[]
    score_F1=[]

    with torch.no_grad():
        for idx in pbar:
            image, label = next(iter_test_dataloader)
            image, label = image.to(device), label.to(device)
            pred = model(image)['out']
            # pred2 = model2(image)['out']
            # pred3 = model3(image)['out']

            B, C, H, W = label.shape
            pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
            # pred2 = F.interpolate(pred2, size=[H, W], mode='bilinear', align_corners=False)
            # pred3 = F.interpolate(pred3, size=[H, W], mode='bilinear', align_corners=False)
            # pred = (pred+pred2+pred3)/3
            pred = torch.softmax(pred, dim=1)
            # pred2 = torch.softmax(pred2, dim=1)
            pred = torch.argmax(pred, dim=1)
            # pred2 = torch.argmax(pred, dim=1)
            # uncertainty_map21 = torch.mean(torch.stack([pred, pred2]), dim=0)
            # uncertainty_map21 = -1.0 * torch.sum(uncertainty_map21*torch.log(uncertainty_map21 + 1e-6), dim=1, keepdim=True)
            # uncertainty_map21 = uncertainty_map21.cpu().detach().numpy()
            
            # pred = torch.argmax(pred, dim=1)
            
            label = label.squeeze(1).long()
            
            predict1 = pred.cpu().detach().numpy()
            
            predict_b = np.where(predict1 >= 0.5, 1, 0)
           
            # predict = pred.cpu().detach().numpy().flatten()
            predict = predict_b.flatten()
            # cv2.imwrite(
            #             # f"./save_picture_drive_withoutbinary/pre{idx}.png", np.uint8(predict_b[0]*255))
            #             f"./save_picture_drive_withoutbinary21/pre{idx}.png", np.uint8(uncertainty_map21[0][0]*255))
           
            gt=label.cpu().detach().numpy()
            gt=np.where(gt >= 0.5, 1, 0)
            target = label.cpu().detach().numpy().flatten()
            tp = (predict * target).sum()
            tn = ((1 - predict) * (1 - target)).sum()
            fp = ((1 - target) * predict).sum()
            fn = ((1 - predict) * target).sum()
            auc = roc_auc_score(target, predict)
            # acc = (tp + tn) / (tp + fp + fn + tn)
            acc =accuracy_score(target, predict)
            pre = tp / (tp + fp)
            sen = tp / (tp + fn)
            spe = tn / (tn + fp)
            iou = tp / (tp + fp + fn)
            f1 = 2 * pre * sen / (pre + sen)
            score_IOU.append(iou)
            score_ACC.append(acc)
            score_AUC.append(auc)
            score_SE.append(sen)
            score_SP.append(spe)
            score_F1.append(f1)
            print("predshape")
            print(pred.size())
            print("labekshape")
            print(label.size())
########################差异像素表示
            img_3=np.zeros((512,512,3), np.uint8)
            img_3[:,:,0]=np.uint8(predict_b[0]*255)
            img_3[:,:,1]=np.uint8(predict_b[0]*255)
            img_3[:,:,2]=np.uint8(predict_b[0]*255)
            # img_3 = cv2.merge([np.uint8(predict_b[0]*255), np.uint8(predict_b[0]*255), np.uint8(predict_b[0]*255)])
            for m in range(predict_b[0].shape[0]):
                for n in range(predict_b[0].shape[1]):
                    if predict_b[0][m][n]==1:
                        # print("predict_b[0][m][n]")
                        # print(predict_b[0][m][n])
                        if gt[0][m][n]==0:
                            # print("gt[0][m][n]")
                            # print(gt[0][m][n])
                            # a1[m][n]==0
                            # a2[m][n]==0
                            # a3[m][n]==255
                            img_3[m][n]=[0,155,0]
                            # a1[i][j]=255#fp
                    if predict_b[0][m][n]==0:
                        if gt[0][m][n]==1:
                            # print("gt[0][m][n]")
                            # print(gt[0][m][n])
                            # img_3[:,:,0][m][n]==0
                            # img_3[:,:,1][m][n]==255
                            # img_3[:,:,2][m][n]==0
                            # a1[m][n]==0
                            # a2[m][n]==255
                            # a3[m][n]==0
                            img_3[m][n]=[255,0,0]
                            # a1[i][j]=255#fn
            # img_3 = cv2.merge([a1*0, a2*0, a3])
            # cv2.imwrite(
            #             f"./save_picture_stare_difference/pre{idx}.png", img_3)



            label_onehot = torch.nn.functional.one_hot(label, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            pred_onehot = torch.nn.functional.one_hot(pred, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            dices = dice_score_batch(prediction=pred_onehot, target=label_onehot).cpu().numpy()
            for b in range(len(dices)):
                for i in range(args.num_classes):
                    results[i].append(dices[b][i])
            print('itr/itrs: {}/{}, label: {}, pred: {}'.format(idx + 1, len(pbar), label.shape, pred.shape))
    # save results
    average_acc=np.mean(score_ACC)
    average_auc=np.mean(score_AUC)
    average_se=np.mean(score_SE)
    average_sp=np.mean(score_SP)
    average_f1=np.mean(score_F1)
    average_iou=np.mean(score_IOU)
    data_frame = pd.DataFrame(
        data={i: results[i] for i in range(args.num_classes)},
        index=range(1, length + 1))
    data_frame.to_csv(project_path + '/' + 'evaluation.csv', index_label='Index')
    result = data_frame.values
    avg_score = np.mean(result, axis=0)
    with open(project_path+'/performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(avg_score[1:]))
    print('AVG Score:', avg_score[1:])
    print('average_acc',average_acc)
    print('average_auc',average_auc)
    print('average_se',average_se)
    print('average_sp',average_sp)
    print('average_f1',average_f1)
    print('average_iou',average_iou)
    
    print('EVAL FINISHED!')


if __name__ == '__main__':
    eval()
