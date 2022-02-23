# deep_parrot

Create a Model that processes Shaspearian text (one dialogue at a time)
and learns to reproduce it, by adding noise in between. The idea is to have
it trained to Generate text as accurately as possible.

# TODO:
- preprocess text, vectorise it, index it, ensure that embeddings and tensors are all ok
- create the model; the idea is to have a BERT -> Attention Layer (Encoder) -> Middle Network -> Attention Layer (Decoder)

# WHY?
I was asked by Luke Robinson to do that, as part of an interview test.
This is by far the most interesting interview test I've had since Huawei's Reinforcement Learning with Attention Layer.

