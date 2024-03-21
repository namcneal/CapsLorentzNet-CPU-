import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from CapsLorentzNet import CapsuleLayer
from LorentzNet import LorentzNet



class CapsuleNetwork(nn.Module):
    def __init__(self,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size,
                 no_scalar,
                 caps_in,
                 no_hidden,
                 dropout_rate,
                 no_layers,
                 c_weight_fact,
                 target_size,
                 dev):
        super(CapsuleNetwork, self).__init__()

        self.net1 = LorentzNet(n_scalar = no_scalar, n_hidden = no_hidden, n_output = caps_in, n_class = num_output_units,
                       dropout = dropout_rate, n_layers = no_layers,
                       c_weight = c_weight_fact)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=caps_in,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False,
                                    in_const=102,
                                    out_const = 36,
                                    out_channels = 32,
                                    devi = dev)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True,
                                   in_const=102,
                                   out_const = 36,
                                   out_channels = 32,
                                   devi = dev)

        
        
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, target_size*4)
        self.reconstruct1 = nn.Linear(target_size*4, target_size*2)
        self.reconstruct2 = nn.Linear(target_size*2, target_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.device = dev
    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        return self.digits(self.primary(self.net1(scalars, x, edges, node_mask, edge_mask, n_nodes)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average),self.margin_loss(input, target, size_average), self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).to(self.device)
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
#        print(T_c.shape)
#        print(max_l.shape)
#        print(max_r.shape)
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size())).to(self.device)
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 1.0

        # Average over batch
        if size_average:
            error = error.mean()

        return error
