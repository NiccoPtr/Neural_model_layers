### Parameters for DA\_Tonic / DA\_Phasic modulation



##### Aim



* **Low DA\_Tonic** --> difficulty to get a clear decision, frequent oscillatory activity in the MC
* **High DA\_Tonic** --> high chances to take a clear decision and maintaining it



* **DA\_Phasic** --> allows MC units to reach saturation and permit a new decision to take place if possible



|**parameters / DA\_lv \& Inp**|inp 0.0 0.0 DA 0.0|inp 0.0 0.0 DA 0.2|inp 0.0 0.0 DA 0.4|inp 0.0 0.0 DA 0.6|inp 0.0 0.0 DA 1.0|
|-|-|-|-|-|-|
|delta\_DLS\_1 (0.8)<br />delta\_DLS\_2 (3.3)<br />Y\_DLS\_1 (0.1)<br />Y\_DLS\_2 (0.2)<br />DLS\_1\_GPi\_W (4.0)<br />DLS\_2\_GPe\_W (3.0)<br />MC\_MGV (2.2)<br />MGV\_MC (1.2)<br />|Struggle to take even one choice for noise|Few choices were taken but could be maintained|Choices are taken and maintained for longer, but the activity is oscillatory and there are switches|Choice is taken and maintained|Saturation|



|**parameters / DA\_lv \& Inp**|inp 0.0 1.0 DA 0.0|inp 0.0 1.0 DA 0.2|inp 0.0 1.0 DA 0.4|inp 0.0 1.0 DA 0.6|inp 0.0 1.0 DA 0.9/1.0|
|-|-|-|-|-|-|
|delta\_DLS\_1 (0.8)<br />delta\_DLS\_2 (3.3)<br />Y\_DLS\_1 (0.1)<br />Y\_DLS\_2 (0.2)<br />DLS\_1\_GPi\_W (4.0)<br />DLS\_2\_GPe\_W (3.2)<br />MC\_MGV (2.2)<br />MGV\_MC (1.2)<br />|Attempt to take few choices with bias present|Choice taken and maintained noisly with bias present|Choice is taken and maintained with bias present|Choice is taken and maintained with bias present|Saturation|

|**parameters / DA\_lv \& Inp**|inp 0.0 1.0 DA 0.0|inp 0.0 1.0 DA 0.2|inp 0.0 1.0 DA 0.4|inp 0.0 1.0 DA 0.6|inp 0.0 1.0 DA 0.9/1.0|
|-|-|-|-|-|-|
|delta\_DLS\_1 (0.6)<br />delta\_DLS\_2 (3.3)<br />Y\_DLS\_1 (0.1)<br />Y\_DLS\_2 (0.2)<br />DLS\_1\_GPi\_W (4.0)<br />DLS\_2\_GPe\_W (3.2)<br />MC\_MGV (2.2)<br />MGV\_MC (1.2)|Attempt to take few choices with bias present|Choice taken but cannot maintain lock-in presenting broad oscillations, with bias|Choice is taken and maintained with bias present|Choice is taken and maintained with bias present|Saturation|



