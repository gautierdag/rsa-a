import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReferentialTrainer(nn.Module):
    def __init__(self, sender, receiver):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        batch_size, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=device).expand(
            len(seq_lengths), max_len
        ) < seq_lengths.unsqueeze(1)

        if self.training:
            mask = mask.type(dtype=messages.dtype)
            messages = messages * mask.unsqueeze(2)
            # give full probability (1) to eos tag (used as padding in this case)
            messages[:, :, self.sender.pad_id] += (mask == 0).type(dtype=messages.dtype)
        else:
            # fill in the rest of message with eos
            messages = messages.masked_fill_(mask == 0, self.sender.pad_id)

        return messages

    def forward(self, target, distractors):
        batch_size = target.shape[0]

        target = target.to(device)
        distractors = [d.to(device) for d in distractors]

        messages, lengths, entropy, h_s, h_rnn_s = self.sender(target)
        messages = self._pad(messages, lengths)
        h_r, h_rnn_r = self.receiver(messages=messages)

        target = target.view(batch_size, 1, -1)
        r_transform = h_r.view(batch_size, -1, 1)

        target_score = torch.bmm(target, r_transform).squeeze()  # scalar

        all_scores = torch.zeros((batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        loss = 0
        for i, d in enumerate(distractors):
            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            all_scores[:, i + 1] = d_score
            loss += torch.max(
                torch.tensor(0.0, device=device), 1.0 - target_score + d_score
            )

        # Calculate accuracy
        all_scores = torch.exp(all_scores)
        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        if self.training:
            return (torch.mean(loss), torch.mean(accuracy), messages)
        else:
            return (
                torch.mean(loss),
                torch.mean(accuracy),
                entropy,
                messages,
                h_s.detach(),
                h_rnn_s.detach(),
                h_r.detach(),
                h_rnn_r.detach(),
            )
