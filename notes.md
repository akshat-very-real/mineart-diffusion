# MineArt - ML Theory Notes

## 1. Diffusion Model

A diffusion model is a generative model that learns to create images by gradually removing noise.

During training, noise is added to real Minecraft images. The model learns to predict the noise.

During generation, the process is reversed: the model starts with random noise and gradually denoises it into a Minecraft image.

---

## 2. Forward Diffusion

The forward process gradually adds random noise to an image.

The image becomes increasingly noisy as the timestep increases.

- Original image -> slightly noisy
- Slightly noisy -> more noisy
- More noisy -> almost pure noise

---

## 3. Reverse Diffusion

The model learns to reverse the noise process.

It starts with random noise and repeatedly predicts and removes noise until a final image is produced.

This process is called sampling or reverse diffusion.

---

## 4. U-Net

U-Net is the neural network used by the diffusion model.

Its main task is to predict the noise present in a noisy image.

It learns visual features such as:

- Edges
- Shapes
- Textures
- Colors
- Minecraft block patterns
- Terrain structures

---

## 5. Loss Function

Loss measures how different the model's prediction is from the correct target.

For diffusion, the predicted noise is compared with the actual noise that was added.

A common loss function is Mean Squared Error (MSE).

Lower loss generally means the model is performing better on its training objective.

However, the goal is not necessarily to make the loss exactly 0, because very low training loss can indicate overfitting.

---

## 6. Backpropagation

Backpropagation determines how the model's parameters contributed to the error.

It calculates gradients from the loss.

The optimizer then uses these gradients to update the model's parameters so that future predictions become better.

---

## 7. Weights

Weights are learned numerical parameters in a neural network.

They determine how strongly different inputs and features influence the model's output.

During training, the weights are continuously adjusted to improve the model's predictions.

---

## 8. Bias

Bias is another learned parameter that shifts the output of a neuron.

A simple neuron can be represented as:

`output = (weight × input) + bias`

Both weights and biases are learned during training.
