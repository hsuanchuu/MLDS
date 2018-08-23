import numpy as np
import tensorflow as tf
import random
import os
from agent_dir.agent import Agent
from collections import deque

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
INITIAL_REPLAY_SIZE = 50000
EXPLORATION_STEPS = 1e6
NUM_REPLAY_MEMORY = 100000
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 5000

seed = 11037
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        print("build model architecture...")
        self.env = env
        self.args = args
        self.state_size = (84, 84, 4)
        self.n_actions = self.env.action_space.n
        self.gamma = 0.99
        self.epsilon = INITIAL_EPSILON
        #inital epsilon - final epsilon
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        self.total_reward, self.total_q_max, self.total_loss, self.duration, self.episode = 0, 0, 0, 0, 0
        self.t = 0
        self.avg_score = []
        self.train_summary = []

        self.replay_memory = deque()

        self.build_net()
        t_params = tf.get_collection('tar_net_params')
        self.e_params = tf.get_collection('eval_net_params')
        self.update_tar_net = [tf.assign(t, e) for t, e in zip(t_params, self.e_params)]

        optimizer = tf.train.RMSPropOptimizer(self.args.lr, momentum=0, epsilon=1e-6, decay=0.99)
        #gradient clipping
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads = [gv[0] for gv in grads_and_vars]
        params = [gv[1] for gv in grads_and_vars]
        # print(grads)
        grads = tf.clip_by_global_norm(grads, 1)[0]

        grads = [grad for grad in grads if grad != None]

        self.train_op = optimizer.apply_gradients(zip(grads, params))
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(self.e_params, max_to_keep=20)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_tar_net)

        self.model_dir = 'save/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if args.train_dqn:
            self.log_dir = 'log/'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if args.test_dqn:
            #you can load your model here
            model_path = 'model.ckpt-38000'
            self.saver.restore(self.sess, self.model_dir + model_path)
            print('loading trained model')

        

    def build_net(self):
        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

        def build_layers(s, name):
            w_init = tf.truncated_normal_initializer(0, 1e-2)
            b_init = tf.constant_initializer(1e-2)
            with tf.variable_scope('conv1'):
                w_conv1 = tf.get_variable('w_conv1',[8,8,4,32],initializer=w_init,collections=name)
                b_conv1 = tf.get_variable('b_conv1',[32],initializer=b_init,collections=name)
                conv1 = conv2d(s, w_conv1, 4)
                h_conv1= tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))

            with tf.variable_scope('conv2'):
                w_conv2 = tf.get_variable('w_conv2',[4,4,32,64],initializer=w_init,collections=name)
                b_conv2 = tf.get_variable('b_conv2',[64],initializer=b_init,collections=name)
                conv2 = conv2d(h_conv1, w_conv2, 2)
                h_conv2= tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))

            with tf.variable_scope('conv3'):
                w_conv3 = tf.get_variable('w_conv3',[3,3,64,64],initializer=w_init,collections=name)
                b_conv3 = tf.get_variable('b_conv3',[64],initializer=b_init,collections=name)
                conv3 = conv2d(h_conv2, w_conv3, 1)
                h_conv3= tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))
                h_conv3_flat = tf.reshape(h_conv3, [-1,3136])

            with tf.variable_scope('fc1'):
                w1 = tf.get_variable('w1', [3136,512], initializer=w_init, collections=name)
                b1 = tf.get_variable('b1', [512], initializer=b_init, collections=name)
                fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_conv3_flat, w1), b1))

            if self.args.duel:
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [512,1], initializer=w_init, collections=name)
                    b2 = tf.get_variable('b2', [1], initializer=b_init, collections=name)
                    V = tf.nn.bias_add(tf.matmul(fc1, w2), b2)

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [512,self.n_actions], initializer=w_init, collections=name)
                    b2 = tf.get_variable('b2', [self.n_actions], initializer=b_init, collections=name)
                    A = tf.nn.bias_add(tf.matmul(fc1, w2), b2)

                with tf.variable_scope('Q'):
                    Q = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [512,self.n_actions], initializer=w_init, collections=name)
                    b2 = tf.get_variable('b2', [self.n_actions], initializer=b_init, collections=name)
                    Q = tf.nn.bias_add(tf.matmul(fc1, w2), b2)

            return Q

        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], name='state')
        self.y = tf.placeholder(tf.float32, [None])
        with tf.variable_scope('eval_net'):
            name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_layers(self.s, name)

        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name='s_')
        with tf.variable_scope('tar_net'):
            name = ['tar_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_target = build_layers(self.s_, name)

        with tf.name_scope('optimization_step'):
            self.action_mask = tf.placeholder(tf.float32, [None, self.n_actions])
            self.q_value = tf.reduce_sum(tf.multiply(self.q_eval, self.action_mask), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.q_value))
            '''self.error = self.y - self.q_value
            quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
            linear_part = self.error - quadratic_part
            self.loss = tf.reduce_mean(tf.sqrt(1 + tf.square(quadratic_part)) - 1 + linear_part, axis=-1)'''

            tf.summary.scalar('Loss', self.loss)

            


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.t = 0


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.env.seed(seed)
        train_t = 0
        while train_t < 5000000:
            state = self.env.reset()
            done = False
            #for _ in range(random.randint(1, 30)):
            #    state, reward, done, _ = self.env.step(0)

            
            while not done:
                
                if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                    action = random.randrange(self.n_actions)
                    mode = 0
                else:
                    actions_val = self.sess.run(self.q_eval,
                                           feed_dict = {self.s:[state]})
                    action = np.argmax(actions_val)
                    mode = 1

                if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
                    self.epsilon -= self.epsilon_step

                next_state, reward, done, _ = self.env.step(action)
                self.train_net(state, action, reward, done, next_state)
                state = next_state

            train_t += 1


    def train_net(self, state, action, reward, terminal, next_state):
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        loss = 0
        if self.t >= INITIAL_REPLAY_SIZE:
            if self.t % TRAIN_INTERVAL == 0:
                #train network
                state_batch = []
                action_batch = []
                reward_batch = []
                next_state_batch = []
                terminal_batch = []
                #sample ramdom minibatch from replay memory
                minibatch = random.sample(self.replay_memory, self.args.batch_size)
                for data in minibatch:
                    state_batch.append(data[0])
                    action_batch.append(data[1])
                    reward_batch.append(data[2])
                    next_state_batch.append(data[3])
                    terminal_batch.append(data[4])

                action_input = np.zeros((self.args.batch_size, self.n_actions))
                for i in range(self.args.batch_size):
                    action_input[i][action_batch[i]] = 1

                y_batch = []
                if self.args.double:
                    #print("double DQN")
                    q_batch_now = self.sess.run(self.q_eval,
                                                feed_dict={self.s:next_state_batch})
                    q_batch = self.sess.run(self.q_target, 
                                            feed_dict={self.s_:next_state_batch})
                    for i in range(self.args.batch_size):
                        done = terminal_batch[i]
                        if done:
                            y_batch.append(reward_batch[i])
                        else:
                            double_q = q_batch[i][np.argmax(q_batch_now[i])]
                            y = reward_batch[i] + self.gamma * double_q
                            y_batch.append(y)

                else:
                    q_batch = self.sess.run(self.q_target, 
                                            feed_dict={self.s_:next_state_batch})
                    for i in range(self.args.batch_size):
                        done = terminal_batch[i]
                        if done:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + self.gamma * np.max(q_batch[i]))

                q_eval, _, loss = self.sess.run([self.q_eval, self.train_op, self.loss],
                                                feed_dict={self.s:state_batch,
                                                           self.y:np.asarray(y_batch),
                                                           self.action_mask:action_input})
                self.total_loss += loss 

            #update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_tar_net)

            self.total_reward += reward

            state = np.expand_dims(state, axis=0)
            self.total_q_max += np.max(self.sess.run(self.q_eval, feed_dict={self.s:state}))


            self.duration += 1

            if terminal:
                if len(self.avg_score) < 30:
                    self.avg_score.append(self.total_reward)
                else:
                    index = self.episode % 30
                    self.avg_score[index] = self.total_reward

                #save network
                if self.episode > 10000 and self.episode % 500 == 0:
                    save_path = self.saver.save(self.sess, self.model_dir+'model.ckpt', global_step=self.episode)
                    print("successfully saved:" + save_path)

                #print information
                if self.t < INITIAL_REPLAY_SIZE:
                    mode = 'rand'
                elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                    mode = 'explore'
                else:
                    mode = 'exploit'
                print('Episode: {0:6d}, T: {1:8d}, Step: {2:5d}, Epsilon: {3:.5f}, Score: {4:3.0f}, Avg_score: {5:3.3f}, mode: {6}\n'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, 
                np.mean(self.avg_score),
                mode))

                summary = tf.Summary(value=[tf.Summary.Value(tag="Average_Rewards", simple_value=np.mean(self.avg_score))])
                
                self.writer.add_summary(summary, self.episode)

                with open('info.txt', 'a') as file:
                    file.write(str(np.mean(self.avg_score)) + '\n')

                with open('loss.txt', 'a') as file:
                    file.write(str(loss) + '\n')

                self.total_reward, self.total_q_max, self.total_loss, self.duration = 0, 0, 0, 0 
                self.episode += 1

        self.t += 1


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.t += 1
        if self.t > 2500:
            return self.env.get_random_action()
        else:
            state = np.expand_dims(observation, axis=0)
            action = np.argmax(self.sess.run(self.q_eval, feed_dict={self.s:state}))
            return action
