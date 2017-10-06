import prune_reward_regressor
import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

rew_reg = prune_reward_regressor.PruneRewardRegressor(session=session)

fake_in_data = np.random.random_sample(size=(25,1))
fake_accuracy_drop = np.random.random_sample(size=(25,1))
fake_out = []
for s_in, s_acc in zip(fake_in_data.ravel().tolist(),fake_accuracy_drop.ravel().tolist()):

    if s_acc>0.0:
        fake_out.append(np.log(1+s_in)*s_acc)
    else:
        fake_out.append(np.log(1+(1.0-s_in))*s_acc)

fake_out_data = np.asarray(fake_out).reshape(-1,1)

for step in range(5):

    batch_ind = np.random.randint(0,25,size=(10))
    rew_reg.train_mlp_with_data(fake_in_data[batch_ind,],fake_out_data[batch_ind,])

    print(rew_reg.predict_best_prune_factor())

