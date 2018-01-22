import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

import matplotlib.pyplot as plt
%matplotlib inline

from IPython import display
import gym
from envs import make_env
from subproc_vec_env import SubprocVecEnv
from model import CNNPolicy
from storage import RolloutStorage
import time

from argument import Args

# update the oldest obs in current_obs by obs

    
def main():
    envs = [make_env(env_name, seed, rank, log_dir) for rank in range(num_processes)]
    envs = SubprocVecEnv(envs)
    obs_shape = envs.observation_space.shape
    obs_shape = [obs_shape[0]*num_stack, *obs_shape[1:]]
    actor_critic = CNNPolicy(obs_shape[0], envs.action_space, False)
    if cuda:
        actor_critic.cuda()
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(num_processes, *obs_shape)
    
    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
            current_obs[:, -shape_dim0:] = obs
            
            obs = envs.reset()
            
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)
    episode_rewards = torch.zeros([num_processes,1])
    final_rewards = torch.zeros([num_processes,1])
    if cuda:
        rollouts.cuda()
        current_obs = current_obs.cuda()
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]
        
        # test
    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                  Variable(rollouts.states[step], volatile=True),
                                                                  Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze().cpu().numpy()
            #print(cpu_action)

            # obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            # stack: make sure that reward is a numpy array(convert list to ndarray)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            # update obs nad rollouts
            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        # compute current update's return
        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, False, gamma, tau)

        # in a2c the values  were calculated twice
        # the data in rollouts must be viewed, because the shape in rollouts is [num_steps, num_processes, x] which is [num,x] in actor_critic
        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),                                                                                       Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                 Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                       Variable(rollouts.actions.view(-1, action_shape)))

        # compute the loss
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        # update model
        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef 
        loss.backward()
        nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
        optimizer.step()

        rollouts.after_update()
        if j % log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * num_processes * num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
            format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    final_rewards.mean(),
                    final_rewards.median(),
                    final_rewards.min(),
                    final_rewards.max(), dist_entropy.data[0],
                    value_loss.data[0], action_loss.data[0]))
# todo: test save_url                
    torch.save(actor_critic,save_url)      

    
def __name__ == '__main__':
    main()


