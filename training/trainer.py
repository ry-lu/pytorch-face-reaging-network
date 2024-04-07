import torch
from torch.nn.modules import BCEWithLogitsLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from piq import LPIPS
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic losses
adversarial_loss = BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
perceptual_loss = LPIPS().to(device)

# Default loss weights
lambda_l1 = 1
lambda_perceptual = 1
lambda_adversarial = 0.05

# Basic pl lightning module for checkpointing and logging
class FRAN(pl.LightningModule):
  def __init__(self, generator, discriminator):
    super(FRAN, self).__init__()
    self.generator = generator
    self.discriminator =discriminator
    self.automatic_optimization = False
    self.my_step =0

  def forward(self, x):
      with torch.no_grad():
        model_output = self.generator(x)
      return model_output

  def training_step(self, batch, batch_idx):
      opt_g, opt_d = self.optimizers()

      opt_g.zero_grad()
      opt_d.zero_grad()


      inputs = batch['input'].to(self.device)
      normalized_input_image = batch['normalized_input_image'].to(self.device)
      normalized_target_image = batch['normalized_target_image'].to(self.device)


      target_age = batch['target_age'].to(self.device)
      # Forward pass
      outputs = self.generator(inputs)

      predicted_images = normalized_input_image+outputs
      predicted_images_with_age = torch.cat((predicted_images, target_age), dim=1)

      real_labels = torch.ones(inputs.shape[0], 1, 32, 32).to(self.device)
      fake_labels = torch.zeros(inputs.shape[0], 1, 32, 32).to(self.device)

      # Compute discriminator losses
      real_loss = adversarial_loss(self.discriminator(torch.cat((normalized_target_image, target_age), dim=1)), real_labels)
      fake_loss = adversarial_loss(self.discriminator(predicted_images_with_age.detach()), fake_labels)

      d_loss = (real_loss+fake_loss) / 2

      self.manual_backward(d_loss)
      opt_d.step()

      # Compute generator loss
      l1_loss_value = l1_loss(predicted_images, normalized_target_image)
      perceptual_loss_value = perceptual_loss(predicted_images, normalized_target_image)
      adversarial_loss_value = adversarial_loss(self.discriminator(predicted_images_with_age), real_labels)

      total_loss =  lambda_adversarial*adversarial_loss_value+ lambda_perceptual*perceptual_loss_value + lambda_l1*l1_loss_value

      self.manual_backward(total_loss)
      opt_g.step()

      # Log loss
      self.log('fake_loss', fake_loss, prog_bar=True)
      self.log('discriminator_loss', d_loss, prog_bar=True)
      self.log('total_loss', total_loss, prog_bar=True)
      self.log('gen_adversarial_loss', lambda_adversarial*adversarial_loss_value, prog_bar=True)
      self.log('perceptual_loss', perceptual_loss_value.mean()*lambda_perceptual, prog_bar=True)
      self.log('l1_loss', l1_loss_value*lambda_l1, prog_bar=True)

      # Display images every 500 steps
      if self.my_step % 500 == 0:

        sample_image = inputs[0]
        model_output = outputs.detach()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(((normalized_input_image[0].cpu().permute(1,2,0).numpy()+1)*127.5).astype('uint'))
        plt.title(f"Input Image, Age: {int(sample_image[3][0][0]*100)}")
        plt.axis('off')

        plt.subplot(1, 5, 2)
        plt.imshow(((normalized_target_image[0].cpu().permute(1,2,0).numpy()+1)*127.5).astype('uint'))
        plt.title("Target Image")
        plt.axis('off')

        new_image = (((normalized_input_image[0].cpu()+model_output[0].cpu()).permute(1,2,0).numpy()+1)*127.5).astype('uint')
        plt.subplot(1, 5, 3)
        plt.imshow(new_image)
        plt.title(f"Output Image, Age: {int(sample_image[4][0][0]*100)}")
        plt.axis('off')

        new_image = (((torch.abs(normalized_target_image[0].cpu()-normalized_input_image[0].cpu())).permute(1,2,0).numpy()+1)*127.5).astype('uint')
        plt.subplot(1, 5, 4)
        plt.imshow(new_image)
        plt.title(f"Tar RGB Diff")
        plt.axis('off')

        new_image = (((torch.abs(model_output[0].cpu())).permute(1,2,0).numpy()+1)*127.5).astype('uint')
        plt.subplot(1, 5, 5)
        plt.imshow(new_image)
        plt.title(f"Pred RGB Diff")
        plt.axis('off')

        plt.show()

      self.my_step +=1

  def configure_optimizers(self):
    generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
    discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

    return [generator_optimizer, discriminator_optimizer]