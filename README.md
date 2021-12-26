# MusicVAE
***
## Music_VAE_Drum_TF2 implement
1. Preprocess
    - midi file (pretty_midi) -> numpy -> tf.data -> Tfrecord로 변환
    - Midi 파일에서 ``` notes{pitch: 70, velocity: 80, start_time: 1.2, end_time: 1.5} ``` 정보를 추출
    - 4마디 / 16 step / 1마디 = 길이 64
    - Model input data shape : ``` (512, 64, 4) # batch_size, seq_len, n_feature ```
    - Details
        1. midi file을 pretty midi로 읽고 numpy array화 한 후
        2. info.csv 에 따라서 train val test 로 분리(Data Generator 사용)
        3. tf data dataset from generator
        4. Tfrecord로 변환


2. Model
    - Encoder
        - Bi-LSTM (일반적인 양방향 LSTM)_2048 hidden node * 1
        - stack_bidirectional_dynamic_rnn : keras.layers.Bidirectional(keras.layers.RNN(cell))
        - stack-Bi-dynamic_rnn의 출력에서 concatenate([Last_forward_Hidden_State, backward_Hidden_State], axis=1) 을 리턴  -> Linear Layer에서 mu, std, norm 추출해서 re-parameterize

    - Latent Vector size 512

    - Decoder
        - Categorical-LSTM-Decoder (한가지 카테고리 출력)_2048 hidden node * 3
        - sampling : 카테고리 분포(1 ~ n 까지의 int 중 하나가 나오는 확률 변수 분포), inverse sigmoid 로 스케줄링

    - 학습
        - 최적화 함수 : Adam | lr annealing 10^-3 ~ 10^-5 decay rate 0.9999
        - 오차 함수 : ELBO = ((flat vector의 argmax 값을 accuracy | softmax cross entropy with logits(스케줄된 샘플링 값)) + KL-divergence)
        - epoch 50 ~ 100