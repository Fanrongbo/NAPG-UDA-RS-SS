import numpy as np
import torch
def convert_to_onehot(sca_label, class_num=2):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=2):
        class_num=2
        # print('s_label, t_label', s_label.shape, t_label.shape)
        # print(t_label)
        #t_label [bs,class] s_label[bs]
        # print('s_label',s_label,t_label)
        batch_size = s_label.size()[0]

        s_sca_label = s_label.cpu().data.numpy()#[bs]  what the class num:1,2,....,class
        s_sca_label=s_sca_label.astype(np.int)
        s_vec_label = convert_to_onehot(s_sca_label)#[bs,class]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)#Count the number of occurrences of each class

        # print('s_vec_label', s_vec_label)
        s_sum[s_sum == 0] = 100
        # print('s_sum',s_sum)
        s_vec_label = s_vec_label / s_sum#In a bacth, the sum of probabilities of category is guaranteed to be 1 or 0;
        # print('t_label',t_label.shape)
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()#[bs]  what the class num:1,2,....,class

        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()#[bs,class_num] the probability of each classes

        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)#the sum of probability of each classes

        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum#[1,class_num]
        # print('t_sca_label', t_sca_label)
        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)#delect the duplicates
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):#loop all categories
            if i in set_s and i in set_t:
                if s_vec_label[:, i].shape[0]==batch_size and t_vec_label[:, i].shape[0]==batch_size:
                    # print('l')
                    # print('s_vec_label', s_vec_label, 't_vec_label',t_vec_label)
                    s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                    t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                    # print('t_tvec',t_tvec.shape)
                    # print('s_tvec', s_tvec, 't_tvec', t_tvec)
                    ss = np.dot(s_tvec, s_tvec.T)
                    # print('ss',i,ss)
                    weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                    tt = np.dot(t_tvec, t_tvec.T)
                    weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                    # print('tt', i, tt)
                    st = np.dot(s_tvec, t_tvec.T)
                    # print('st',st)
                    weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                    count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        # print('weight_ss',weight_ss)
        # print('weight_tt',weight_tt)
        # print('weight_st',weight_st)


        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
