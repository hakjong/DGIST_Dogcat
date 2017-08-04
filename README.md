# 2017 DGIST 딥러닝 기반 강아지 고양이 이미지 인식 경진대회 소스코드  
[대회 사이트](http://dgist.imagechallenge.kr/)

경북대학교 컴퓨터학부 손학종
* 446898 step 의 checkpoint 를 포함하고 있습니다.

## 설명 & 사용법

* preprocessing.ipynb : 트레이닝 데이터셋을 생성합니다.
    * Jupyter notebook 으로 실행합니다.
    * ./data_train/Dog 에 개 이미지를 넣습니다.
    * ./data_train/Cat 에 고양이 이미지를 넣습니다.
    * ./bin 디렉토리에 xx.bin(train data), test.bin(validation data) 를 생성합니다.
     
* test_preprocessing.ipynb : 테스트 데이터셋을 생성합니다.
    * Jupyter notebook 으로 실행합니다.
    * ./data_test 에 테스트 이미지들을 넣습니다.
    * ./bin 디렉터리에 tocsv.bin(test data) 를 생성합니다.
    
* dogcat_train.py : 트레이닝을 합니다.
    * CMD: python3 dogcat_train.py
    * ./train 에 체크포인트 파일을 생성합니다.
    * !! 실행 시 ./train 디렉토리를 삭제합니다.
    * 990000 batch step 만큼 학습합니다.
    * GPU 메모리의 50%만 사용합니다.
    

* dogcat_eval.py : 체크포인트 파일이 생성될 때 마다 validation data 로 평가하고 기록합니다.
    * CMD: python3 dogcat_eval.py
    * ./eval.log 에 기록합니다.
    * 트레이닝과 동시에 실행합니다.
    * GPU 메모리의 40%만 사용합니다.

* dogcat_eval2csv.py : 학습 된 모델로 test set 의 추론 결과를 csv 파일로 출력합니다.
    * CMD: python3 dogcat_eval2csv.py
    * ./result.csv 파일을 생성합니다.
    * ./train/checkpoint 파일의 model_checkpoint_path 에 지정된 checkpoint 를 사용합니다.
    * GPU 메모리의 40%만 사용합니다.

* trainbackup.sh : 새로운 checkpoint 가 생성될 때 마다 백업합니다.
    * CMD: ./trainbackup.sh
    * ./train.bak 에 백업합니다.

* Model.pptx : 사용한 모델에 대한 설명입니다.

* dogcat.py : 모델에 대한 정의입니다.

* dogcet_input.py : 이미지 데이터를 읽어오기 위한 모듈입니다.


## License

MIT License  
Copyright (c) 2017 Hakjong Son

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.


## Open source license

TensorFlow Tutorial Models for cifar10  
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10  
  
==========================================================  
Copyright 2015 The TensorFlow Authors. All Rights Reserved.  
  
Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  
  
http://www.apache.org/licenses/LICENSE-2.0  
  
Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
