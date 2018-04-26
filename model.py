import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.contrib import rnn


def make_encoder(segment, batch_size, num_latents, encoder_units):
    with tf.name_scope("encoder", [segment]):
        encoder_cell = rnn.BasicLSTMCell(encoder_units, name="encoder_cell")
        initial_state = encoder_cell.zero_state(batch_size, tf.float32)

        encoder_outputs, final_state = tf.nn.dynamic_rnn(
                encoder_cell, segment,
                initial_state=initial_state)
        q_params = final_state.h

        q_loc = tf.layers.dense(q_params, num_latents, activation=None)
        q_scale = tf.exp(
                tf.layers.dense(q_params, num_latents, activation=None))

        q = tfd.MultivariateNormalDiag(loc=q_loc, scale_diag=q_scale)
    return q, encoder_outputs


def make_decoder(z, batch_size, num_latents, decoder_units,
                 segment_length, segment_channels):
    with tf.name_scope("decoder", [z]):
        decoder_cell = rnn.BasicLSTMCell(decoder_units, name="decoder_cell")
        initial_state = decoder_cell.zero_state(batch_size, tf.float32)
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, segment_length, 1])

        p_params, _ = tf.nn.dynamic_rnn(
                decoder_cell, z_repeated, initial_state=initial_state)

        p_loc = tf.layers.dense(p_params, segment_channels, activation=None)
        p_scale = tf.exp(
                tf.layers.dense(p_params, segment_channels, activation=None))

        p = tfd.MultivariateNormalDiag(loc=p_loc, scale_diag=p_scale)
    return p


def make_prior(num_latents):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(num_latents, tf.float32))


def make_discriminator(z, batch_size, num_units, num_layers):
    with tf.name_scope("make_discriminator", [z]):
        # construct an MLP with logits output
        outputs = z
        for l in range(num_layers):
            activation = tf.nn.leaky_relu if l < (num_layers - 1) else None
            units = num_units if l < (num_layers - 1) else 2
            outputs = tf.layers.dense(outputs, units,
                                      activation=activation,
                                      name="layer_{}".format(l + 1))
    return outputs


def create_vae_with_elbo_loss(
        segment, segment_channels,
        encoder_units, decoder_units, num_latents,
        beta):

    input_attenuation = 100.  # empirical magic number
    segment = segment / input_attenuation

    batch_size, segment_length, _ = tf.unstack(tf.shape(segment))

    q, outs = make_encoder(segment, batch_size, num_latents, encoder_units)
    z = q.sample()
    p = make_decoder(z, batch_size, num_latents, decoder_units,
                     segment_length, segment_channels)
    r = make_prior(num_latents)

    elbo_loss = tf.reduce_mean(
            -tf.reduce_sum(p.log_prob(segment), axis=1) +  # reconstruction loss
            beta * (q.log_prob(z) - r.log_prob(z)))  # KL divergence

    train_vars = tf.trainable_variables()
    l2_losses = [tf.nn.l2_loss(var) for var in train_vars]
    l2_loss = tf.reduce_mean(l2_losses)

    total_loss = elbo_loss + l2_loss * 1e-4

    outputs = {'z_prior': r.sample(batch_size),
               'z_posterior_sample': z,
               'z_posterior_mean': q.mean(),
               'x_reconst_sample': p.sample() * input_attenuation,
               'x_reconst_mean': p.mean() * input_attenuation,
               'encoder_outs': outs}

    return total_loss, outputs


def create_discriminator_with_softmax_loss(
        z_vae, z_disc, batch_size, num_units, num_layers):

    # assume label=0 if for q and label=1 if for q_bar
    labels = tf.concat(
            (tf.tile([[1, 0]], [batch_size, 1]),
             tf.tile([[0, 1]], [batch_size, 1])),
            axis=0)

    # shuffle the batch to get a sample from q_bar instead of q
    z_disc = tf.transpose(tf.map_fn(tf.random_shuffle, tf.transpose(z_disc)))

    with tf.variable_scope("discriminator"):
        logits_vae = make_discriminator(
                tf.stop_gradient(z_vae), batch_size, num_units, num_layers)

    with tf.variable_scope("discriminator", reuse=True):
        logits_disc = make_discriminator(
                tf.stop_gradient(z_disc), batch_size, num_units, num_layers)

    logits = tf.concat((logits_vae, logits_disc), axis=0)
    
    # discriminator training loss
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(labels),
                logits=logits, name="discriminator_loss"))

    # D_KL estimate, i.e. D_KL = log(softmax(class1)/softmax(class2))
    probas = tf.nn.softmax(logits_vae) + 1e-32
    D_KL = tf.reduce_mean(tf.log(probas[:, 0] / probas[:, 1]))

    return D_KL, loss
    


def create_vae_with_factor_loss(batch_vae, batch_disc, segment_channels,
                                encoder_units, decoder_units, num_latents,
                                disc_units, disc_layers,
                                gamma):

    batch_size, segment_length, _ = tf.unstack(tf.shape(batch_vae))

    with tf.variable_scope("encoder"):
        q_vae, outs_vae = make_encoder(
                batch_vae, batch_size, num_latents, encoder_units)
    with tf.variable_scope("encoder", reuse=True):
        q_disc, outs_disc = make_encoder(
                batch_disc, batch_size, num_latents, encoder_units)

    z_vae = q_vae.sample()
    z_disc = q_disc.sample()

    p = make_decoder(z_vae, batch_size, num_latents, decoder_units,
                     segment_length, segment_channels)
    r = make_prior(num_latents)

    D_KL, disc_loss = create_discriminator_with_softmax_loss(
            z_vae, z_disc, batch_size, disc_units, disc_layers)

    factor_loss = (tf.reduce_mean(
            -tf.reduce_sum(p.log_prob(batch_vae), axis=1) +  # reconstruction loss
            (q_vae.log_prob(z_vae) - r.log_prob(z_vae))) +  # KL divergence
            gamma * tf.stop_gradient(D_KL))  # factorizing KL divergence

    train_vars = tf.trainable_variables()
    l2_losses = [tf.nn.l2_loss(var) for var in train_vars]
    l2_loss = tf.reduce_mean(l2_losses)

    vae_loss = factor_loss + l2_loss * 1e-4

    outputs = {'z_prior': r.sample(batch_size),
               'z_posterior_sample': z_vae,
               'z_posterior_mean': q_vae.mean(),
               'x_reconst_sample': p.sample(),
               'x_reconst_mean': p.mean(),
               'encoder_outs': outs_vae}

    return vae_loss, disc_loss, D_KL, outputs
