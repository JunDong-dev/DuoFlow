# Modified for DuoFlow; see NOTICE.md.
"""Regression checks for phoneme padding and blank-token masks."""

import torch

from motionspeech_trainer.phoneme_encoder import PhonemeEncoder


def test_phoneme_encoder():
    """Test phoneme encoder with add_blank support"""
    batch_size = 2
    pad_length = 310
    vocabs_size = 183
    encoder = PhonemeEncoder(vocabs_size=vocabs_size, pad_length=pad_length, pad_token_id=0, out_channels=192, add_blank=False)
    phoneme_ids = torch.randint(1, vocabs_size, (batch_size, pad_length))
    phoneme_ids[:, 200:] = 0
    output = encoder(phoneme_ids)
    embeddings, mask = output
    assert mask.sum(dim=1).tolist() == [200, 200], f'Expected [200, 200], got {mask.sum(dim=1).tolist()}'
    encoder_blank = PhonemeEncoder(vocabs_size=vocabs_size, pad_length=pad_length, pad_token_id=0, out_channels=192, add_blank=True)
    blank_ids = torch.zeros(batch_size, pad_length, dtype=torch.long)
    N = 50
    for i in range(N):
        blank_ids[:, 2 * i + 1] = torch.randint(1, vocabs_size, (batch_size,))
    output_blank = encoder_blank(blank_ids)
    emb_blank, mask_blank = output_blank
    expected_len = 2 * N + 1
    assert mask_blank.sum(dim=1).tolist() == [expected_len, expected_len], f'Expected [{expected_len}, {expected_len}], got {mask_blank.sum(dim=1).tolist()}'
    uncond = encoder_blank.get_unconditional_condition(batch_size)
    uncond_emb, uncond_mask = uncond
    assert uncond_mask.sum(dim=1).tolist() == [0, 0], f'Expected [0, 0], got {uncond_mask.sum(dim=1).tolist()}'
