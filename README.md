# Echo-Debar-AEC
**Submission to the 1st Acoustic Echo Cancellation (AEC) Challenge, organized by Microsoft and ICASSP 2021.

More details about AEC Challenge available here:
https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-icassp-2021/

This repository contains the codes and submission to the AEC challenge by Team Jadavpur University. 


# **Title: Real-time Acoustic Echo Cancellation by Joint Implementation of Adaptive Filter and Deep Neural Network**

**Abstract**: Efficient 'Acoustic Echo Cancellation' (AEC) is a key parameter for satisfactory voice communication, teleconferencing applications. Unlike many signal processing problems that can be addressed by post-processing approaches, AEC algorithms need to work in real-time. In the last few decades, researchers have developed several classical filtering methods to remove the acoustic echoes. Adaptive filtering, which is the most popular of these methods, tries to model the linear echo signal by analysing the near-end microphone and the far-end loopback signal. These classical algorithms are suitable for real-time implementations but suffer from poor performance in residual echo cancellation. In contrast, Deep Neural Networks (DNN) are effective tools to model the non-linear distortions introduced by the echo. However, very dense DNNs are restricted by their ability to process the audio data in real-time for low resource settings. Motivated by these advantages and disadvantages of both approaches, we present here a hybrid echo cancellation algorithm that jointly implements adaptive filtering and deep learning to process the audio data in real-time. The adaptive filter partially models the echo-signal spectrum. Its output is then passed through the DNN to suppress the residual echo, external noise, and other distortions. For this work, we have implemented a network of long-short-term memory (LSTM) units because of its ability to efficiently model sequences of any length. Experimental analysis shows that this joint implementation of the classical and deep-learning approach enables effective acoustic echo-cancellation for both real-time and low-resource applications.


**Team Members**: Annesya Banerjee (Team Lead), Achal Nilhani, Supriya Dhabal.
