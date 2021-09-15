# Detector de Libras

Esse código, faz a interpretação de libras a partir de uma webcam. Atualmente o treinamento contém apenas 10 palavras.

## Contribuidores

<table>
  <tr>
    <td align="center">
        <img src="https://avatars.githubusercontent.com/u/84801416?s=400&v=4" width="80px;" alt="Giovanna"/>
        <br/>
        <sub>
            <b>Giovanna Lima Marques</b>
        </sub>
	</td>
    <td align="center">
		<a href="https://github.com/tiorac">
			<img src="https://avatars.githubusercontent.com/u/1957382?v=4" width="80px;" alt="tiorac"/>
			<br/>
			<sub>
				<b>Ricardo Augusto Coelho</b>
			</sub>
		</a>
	</td>
    <td align="center">
		<a href="https://github.com/tejolteon">
			<img src="https://avatars.githubusercontent.com/u/24478131?v=4" width="80px;" alt="tejolteon"/>
			<br/>
			<sub>
				<b>Tiago Goes Teles </b>
			</sub>
		</a>
	</td>
    <td align="center">
		<a href="https://github.com/wellingtonalb">
			<img src="https://avatars.githubusercontent.com/u/64939751?v=4" width="80px;" alt="wellingtonalb"/>
			<br/>
			<sub>
				<b>Wellington de Jesus Albuquerque </b>
			</sub>
		</a>
	</td>
  </tr>
</table>

## Processo


## Execução

1. Instale as [dependências](#Dependências).
1. Clone o repositório.
    ```cmd
    git clone https://github.com/ia-equipe-6/libras-detector.git
    cd libras-detector
    ```
1. (Opcional) Salve o modelo de treimento na pasta "model".
1. (Opcional) Salve o modelo LabelEncoder com nome "words.encoder.npy".
1. Conecte a WebCam na porta USB do seu computador.
1. Execute o identificador de libras:
    ```cmd
    python .\libras.py
    ```


## Dependências

Esse código foi testado na versão 3.9 do [Python](https://www.python.org/downloads/) e utiliza as seguintes bibliotecas para geração do dataset:

* OpenCV 
    ```cmd
    pip install opencv-python
    ```
* Pandas
    ```cmd
    pip install pandas
    ```
* MediaPipe
    ```cmd
    pip install mediapipe
    ```
* Numpy
    ```cmd
    pip install numpy
    ```
* Scikit-Learn
    ```cmd
    pip install scikit-learn
    ```
* TensorFlow 
    ```cmd
    pip install tensorflow
    ```
* TensorFlow GPU (opcional)
    ```cmd
    pip install tensorflow-gpu
    ```