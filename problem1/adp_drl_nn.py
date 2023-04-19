import numpy as np

import model_def

eta_c_1 = 0.1
eta_a_1 = 0.3
eta_c_2 = 0.01
eta_a_2 = 0.4

zeta_1 = 1.0
zeta_2 = 1.4

class RBFN(object):

    def __init__(self, feature_nums,hidden_nums, output_nums): #还有一些超参数可能需要初始化
        self.hidden_nums = hidden_nums
        self.output_nums = output_nums
        self.feature_nums = feature_nums
        self.sample_nums = 0
        self.gaussian_kernel_width = 0  # 高斯核宽度
        # self.hiddencenters = 72
        self.hiddenoutputs = 0
        self.hiddenoutputs_expand = 0
        self.linearweights = 0
        self.finaloutputs = 0
        # self.gaussian_kernel_width = np.random.random((72, 1))  # 待修改
        self.gaussian_kernel_width = np.linspace(-1, 1, num=72).reshape(72, 1)
        self.hiddencenters = np.linspace(-7.2, 7.2, num=72).reshape(72, 1)
        # self.gaussian_kernel_width =np.zeros((72,1))
        # self.gaussian_kernel_width.fill(0.8)
        # print(self.gaussian_kernel_width)
        # if self.feature_nums == 1:
        #     self.hiddencenters = np.linspace(-7.2, 7.2, num=72).reshape(72, self.feature_nums)
        # if self.feature_nums == 3:
        #     self.hiddencenters = np.linspace(-7.2, 7.2, num=72).reshape(72, self.feature_nums)
        #     self.hiddencenters = np.concatenate(self.hiddencenters,np.linspace(-7.2, 7.2, num=72).reshape(72, self.feature_nums))
        #     self.hiddencenters = np.concatenate(self.hiddencenters,np.linspace(-7.2, 7.2, num=72).reshape(72, self.feature_nums))
        # # self.hiddencenters = np.random.random((self.hidden_nums, self.feature_nums)) # 待修改
        # self.hiddencenters = 14.4 * self.hiddencenters - 7.2
        # print(self.hiddencenters.shape)
        self.linearweights = np.random.random((self.hidden_nums + 1, self.output_nums))
        # self.linearweights = np.zeros((self.hidden_nums + 1, self.output_nums))
        # self.linearweights.fill(0.3)
        # self.linearweights = np.zeros((self.hidden_nums + 1, self.output_nums))

# def init(self):
#         gaussian_kernel_width = np.random.random((self.hiddencenters, 1)) # 待修改
#         # print(gaussian_kernel_width.shape)
#         # print(gaussian_kernel_width.shape)
#         hiddencenters = np.random.random((self.hidden_nums, self.feature_nums))     # 待修改
#         # print(hiddencenters.shape)
#         # print(hiddencenters.shape)
#         linearweights = np.random.random((self.hidden_nums + 1, self.output_nums))                 # 待修改
#         # print(linearweights.shape)
#         # linearweights = np.zeros((self.hidden_nums + 1, self.output_nums))
#         return gaussian_kernel_width, hiddencenters, linearweights

    def forward(self, inputs):
        # print(inputs.shape)
        inputs = inputs.T
        # self.sample_nums, self.feature_nums = inputs.shape
        # print(self.sample_nums)
        # print(self.feature_nums)
        # self.gaussian_kernel_width, self.hiddencenters, self.linearweights = self.init()
        self.hiddenoutputs = self.guass_change(self.gaussian_kernel_width, inputs, self.hiddencenters)
        # print(self.hiddenoutputs)
        self.hiddenoutputs_expand = self.add_intercept(self.hiddenoutputs)
        # print(self.linearweights)
        # print(self.hiddenoutputs_expand)
        self.finaloutputs = np.dot(self.hiddenoutputs_expand, self.linearweights)
        # print(self.finaloutputs)

    def guass_function(self, gaussian_kernel_width, inputs, hiddencenters_i):
        # print(inputs)
        # print(hiddencenters_i)
        # print(gaussian_kernel_width.shape)
        # print(np.linalg.norm((inputs-hiddencenters_i)))
        # print(np.exp(-np.linalg.norm((inputs-hiddencenters_i), axis=1)**2/(2*gaussian_kernel_width**2)))
        return np.exp(-np.linalg.norm((inputs-hiddencenters_i), axis=1)**2/(2*gaussian_kernel_width**2))

    def guass_change(self, gaussian_kernel_width, inputs, hiddencenters):
        hiddenresults = np.zeros((self.sample_nums, len(hiddencenters)))
        # print(hiddenresults.shape)
        for i in range(len(hiddencenters)):
            # print(self.guass_function(gaussian_kernel_width[i], inputs, hiddencenters[i]))
            hiddenresults[:,i] = self.guass_function(gaussian_kernel_width[i],inputs,hiddencenters[i])
            # print(hiddenresults[:,i])
            # print(hiddencenters[i])
        return hiddenresults

    def add_intercept(self, hiddenoutputs):
        return np.hstack((hiddenoutputs, np.ones((self.sample_nums,1))))

class Critic1_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_super = 0
        self.linearweights_last = 0

    def backward(self, certain_model, actor_1_nn):
        # print(self.hiddenoutputs_expand.shape)
        # print(certain_model.a.shape)
        # print(certain_model.z_1.shape)
        # print(actor_1_nn.finaloutputs.shape)
        # print(certain_model.lambda_2.shape)
        self.varpi_super = -self.hiddenoutputs_expand * [
            certain_model.a * zeta_1 * certain_model.z_1 +
            1 / 2 * certain_model.a * actor_1_nn.finaloutputs + certain_model.lambda_2]
        self.varpi_super = self.varpi_super.squeeze(axis = 1)
        self.linearweights_last = self.linearweights
        reg_1 = (-2 * zeta_1 * certain_model.z_1.T * certain_model.lambda_2)
        reg_2 = -((certain_model.a * zeta_1 * zeta_1 - 1) * certain_model.z_1.T * certain_model.z_1)
        reg_3 = (1 / 4 * certain_model.a * actor_1_nn.finaloutputs * actor_1_nn.finaloutputs.T)
        # print(reg_3.shape)
        reg_4 = np.dot(self.varpi_super , self.linearweights_last)
        # print((reg_1 + reg_2 + reg_3 +reg_4).shape)

        reg_sum = (reg_1 + reg_2 + reg_3 +reg_4).squeeze(axis = 1)
        # print(reg_sum)
        # print(self.varpi_super.shape)
        # print(self.varpi_super.T.shape)
        # print(reg_sum.shape)
        # print(np.dot(-eta_c_1 / (1 + np.dot(self.varpi_super , self.varpi_super.T)) , self.varpi_super).shape)
        # print((np.dot(-eta_c_1 / (1 + np.dot(self.varpi_super , self.varpi_super.T)) , self.varpi_super) * reg_sum).shape)

        self.linearweights += (np.dot(-eta_c_1 / (1 + np.dot(self.varpi_super , self.varpi_super.T)) , self.varpi_super) * reg_sum).T

class Actor1_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_super = 0
        self.linearweights_last = 0

    def backward(self, certain_model, critic_1_nn):
        self.varpi_super = -self.hiddenoutputs_expand * [
            certain_model.a * zeta_1 * certain_model.z_1 +
            1 / 2 * certain_model.a * self.finaloutputs + certain_model.lambda_2]
        self.linearweights_last = self.linearweights
        self.varpi_super = self.varpi_super.squeeze(axis=1)
        reg_1 = (1 / 2 * self.hiddenoutputs_expand.T * certain_model.z_1)
        # print(reg_1.shape)
        reg_4 = np.dot(self.hiddenoutputs_expand.T,np.dot(self.hiddenoutputs_expand, self.linearweights_last))
        # print(reg_4.shape)
        # print(np.dot(self.varpi_super , self.varpi_super.T).shape)
        reg_5 = np.dot((eta_c_1 / (4 * (1 + np.dot(self.varpi_super , self.varpi_super.T)))) , reg_4.T)
        # print(reg_5.shape)
        reg_6 = np.dot(self.varpi_super.T , critic_1_nn.linearweights_last.T)
        reg_7 = np.dot(reg_5,reg_6)
        # print(reg_7.shape)
        # reg_2 = ((eta_c_1 / (4 * (1 + np.dot(self.varpi_super , self.varpi_super.T)))) * reg_4 * self.varpi_super.T * critic_1_nn.linearweights_last)
        # print(reg_7.shape)
        reg_3 = (-eta_a_1 * np.dot(self.hiddenoutputs_expand.T , np.dot(self.hiddenoutputs_expand , self.linearweights_last)))
        # print(self.hiddenoutputs_expand.shape)
        # print(np.dot(self.hiddenoutputs_expand.T , self.linearweights_last.T).shape)
        # print(reg_3.shape)
        # print((reg_1 + reg_7.T + reg_3).shape)
        self.linearweights += (reg_1 + reg_7.T + reg_3)

class Critic2_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_sub = 0
        self.linearweights_last = 0

    def backward(self, certain_model, actor_2_nn, a_hat_dot):
        # print(self.hiddenoutputs_expand.shape)
        # print(certain_model.function_V)
        # print(certain_model.z_2.shape)
        # print(actor_2_nn.finaloutputs.T.shape)
        # print(a_hat_dot.shape)
        # print((certain_model.function_V - zeta_2 * certain_model.z_2
        #                       - 1 / 2 * actor_2_nn.finaloutputs.T - a_hat_dot).shape)   # 会变化
        self.varpi_sub = np.dot(-self.hiddenoutputs_expand.T , (certain_model.function_V - zeta_2 * certain_model.z_2
                            - 1 / 2 * actor_2_nn.finaloutputs.T - a_hat_dot).T)
        # print(self.varpi_sub.shape)
        # self.varpi_sub = self.varpi_sub.squeeze(axis=1)
        # print(self.varpi_sub.shape)
        self.linearweights_last = self.linearweights
        reg_1 = 2 * zeta_2 * np.dot(certain_model.z_2.T , (certain_model.function_V - a_hat_dot))
        # print(reg_1.shape)
        reg_2 = - (zeta_2 * zeta_2 - 1) * np.dot(certain_model.z_2.T , certain_model.z_2)
        # print(reg_2.shape)
        # print(actor_2_nn.finaloutputs.shape)
        reg_3 = 1 / 4 * np.dot(actor_2_nn.finaloutputs.T , actor_2_nn.finaloutputs)
        # print(reg_3.shape)
        reg_4 = np.dot(self.varpi_sub.T , self.linearweights_last)
        # print(self.varpi_sub.shape)
        # print(self.linearweights_last.shape)
        #
        # print(reg_4.shape)
        # print((reg_1 + reg_2 + reg_3 +reg_4).shape)
        reg_5 = (reg_1 + reg_2 + reg_3 +reg_4)
        # print((-eta_c_2 / (1 + np.linalg.norm(self.varpi_sub)) * self.varpi_sub).shape)
        # print(self.varpi_sub.shape)
        reg_6 = (-eta_c_2 / (1 + np.linalg.norm(self.varpi_sub)**2) * self.varpi_sub)
        # print(reg_6.shape)
        # self.linearweights += (-eta_c_2 / (1 + np.linalg.norm(self.varpi_sub)) * self.varpi_sub) * reg_5
        self.linearweights += np.dot(reg_6,reg_5)

class Actor2_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_sub = 0
        self.linearweights_last = 0

    def backward(self, certain_model, critic_2_nn, a_hat_dot):
        # print(certain_model.function_V)
        # print(certain_model.z_2)
        # print(critic_2_nn.finaloutputs)
        # print(a_hat_dot)
        self.varpi_sub = np.dot(-self.hiddenoutputs_expand.T, (certain_model.function_V - zeta_2 * certain_model.z_2
                                                               - 1 / 2 * critic_2_nn.finaloutputs.T - a_hat_dot).T)
        # print(self.varpi_sub)
        # print(self.hiddenoutputs_expand.shape)
        # print(self.varpi_sub.shape)
        # 将self.varpi_sub转为（73，1）的数组，可以参考critic_2_a 下一步要做 4.14号
        self.linearweights_last = self.linearweights
        # print(self.varpi_sub.shape)
        # self.varpi_sub = self.varpi_sub.squeeze(axis=1)
        reg_1 = np.dot(1 / 2 * self.hiddenoutputs_expand.T , certain_model.z_2.T)
        # print(reg_1.shape)
        # print(reg_1.shape)
        reg_4 = np.dot(self.hiddenoutputs_expand.T, np.dot(self.hiddenoutputs_expand, self.linearweights_last))
        # print(self.hiddenoutputs_expand.shape)
        # print(reg_4.shape)
        reg_5 = np.dot((eta_c_2 / (4 * (1 + np.linalg.norm(self.varpi_sub)**2))), reg_4.T)
        # print(reg_5.shape)
        reg_6 = np.dot(self.varpi_sub.T, critic_2_nn.linearweights_last)
        reg_7 = np.dot(reg_5.T, reg_6.T)
        # print(reg_7.shape)
        reg_3 = (-eta_a_2 * np.dot(self.hiddenoutputs_expand.T,
                                   np.dot(self.hiddenoutputs_expand, self.linearweights_last)))
        # print(reg_3.shape)
        self.linearweights += (reg_1 + reg_7 + reg_3)
        # print(self.linearweights)