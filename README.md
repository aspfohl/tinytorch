# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```

Copied from [module-3](https://github.com/Cornell-Tech-ML/mle-module-3-aspfohl)
```bash
cp ../mle-module-3-aspfohl/minitorch/tensor_data.py minitorch/tensor_data.py
cp ../mle-module-3-aspfohl/minitorch/tensor_functions.py minitorch/tensor_functions.py
cp ../mle-module-3-aspfohl/minitorch/tensor_ops.py minitorch/tensor_ops.py
cp ../mle-module-3-aspfohl/minitorch/fast_ops.py minitorch/fast_ops.py
cp ../mle-module-3-aspfohl/minitorch/cuda_ops.py minitorch/cuda_ops.py
cp ../mle-module-3-aspfohl/minitorch/operators.py minitorch/operators.py
cp ../mle-module-3-aspfohl/minitorch/module.py minitorch/module.py
cp ../mle-module-3-aspfohl/minitorch/autodiff.py minitorch/autodiff.py
cp ../mle-module-3-aspfohl/minitorch/scalar.py minitorch/scalar.py
cp ../mle-module-3-aspfohl/minitorch/module.py minitorch/module.py
cp ../mle-module-3-aspfohl/project/run_manual.py project/run_manual.py
cp ../mle-module-3-aspfohl/project/run_scalar.py project/run_scalar.py
cp ../mle-module-3-aspfohl/project/run_tensor.py project/run_tensor.py
cp ../mle-module-3-aspfohl/project/parallel_check.py project/parallel_check.py
cp ../mle-module-3-aspfohl/project/run_fast_tensor.py project/run_fast_tensor.py
cp ../mle-module-3-aspfohl/tests/test_autodiff.py tests/test_autodiff.py
cp ../mle-module-3-aspfohl/tests/test_module.py tests/test_module.py
cp ../mle-module-3-aspfohl/tests/test_scalar.py tests/test_scalar.py
cp ../mle-module-3-aspfohl/tests/test_tensor.py tests/test_tensor.py
cp ../mle-module-3-aspfohl/tests/test_tensor_data.py tests/test_tensor_data.py
cp ../mle-module-3-aspfohl/tests/test_tensor_general.py tests/test_tensor_general.py
cp ../mle-module-3-aspfohl/tests/test_operators.py tests/test_operators.py
cp ../mle-module-3-aspfohl/Makefile Makefile
cp ../mle-module-3-aspfohl/poetry.lock poetry.lock
cp ../mle-module-3-aspfohl/pyproject.toml pyproject.toml
mkdir .vscode
cp ../mle-module-3-aspfohl/.vscode/settings.json .vscode/settings.json
echo "" >> .gitignore
echo "# ignore env" >> .gitignore
echo ".vscode" >> .gitignore
echo "poetry.lock" >> .gitignore
echo "pyproject.toml" >> .gitignore
echo "Makefile" >> .gitignore
```

## Task 4.5

MNIST

<img src="img/mnist_layer.png" width="500px">
<img src="img/mnist_loss.png" width="500px">

```bash
Epoch: 1/500, loss: 142.81171980966207, correct: 2
Epoch: 2/500, loss: 142.4754418698712, correct: 4
Epoch: 3/500, loss: 141.45392542293123, correct: 4
Epoch: 4/500, loss: 135.5547345184993, correct: 8
Epoch: 5/500, loss: 102.4022399880563, correct: 13
Epoch: 6/500, loss: 76.99674193452694, correct: 14
Epoch: 7/500, loss: 58.816383680137555, correct: 13
Epoch: 8/500, loss: 46.34509006041616, correct: 13
Epoch: 9/500, loss: 43.031348222166145, correct: 13
Epoch: 10/500, loss: 36.6132564173352, correct: 13
Epoch: 11/500, loss: 32.717989675366816, correct: 13
Epoch: 12/500, loss: 32.509173502674074, correct: 14
Epoch: 13/500, loss: 28.57809294114041, correct: 13
Epoch: 14/500, loss: 25.595939198735998, correct: 13
Epoch: 15/500, loss: 24.492104351002947, correct: 13
Epoch: 16/500, loss: 24.610730077956667, correct: 13
Epoch: 17/500, loss: 22.02278308988139, correct: 13
Epoch: 18/500, loss: 23.518071993521072, correct: 14
Epoch: 19/500, loss: 19.954623806232867, correct: 14
Epoch: 20/500, loss: 21.088707593530067, correct: 14
Epoch: 21/500, loss: 19.36753912852229, correct: 14
Epoch: 22/500, loss: 19.13454483785838, correct: 14
Epoch: 23/500, loss: 16.772193129338927, correct: 13
Epoch: 24/500, loss: 16.889645499094364, correct: 14
Epoch: 25/500, loss: 16.993800944790376, correct: 14
Epoch: 26/500, loss: 17.057327583729702, correct: 14
Epoch: 27/500, loss: 17.917941550893843, correct: 15
Epoch: 28/500, loss: 18.197948562707357, correct: 14
Epoch: 29/500, loss: 17.070905078441296, correct: 14
Epoch: 30/500, loss: 16.674232986423434, correct: 15
Epoch: 31/500, loss: 17.78657872628639, correct: 13
Epoch: 32/500, loss: 14.68719650493648, correct: 15
Epoch: 33/500, loss: 15.672684159129794, correct: 14
Epoch: 34/500, loss: 17.948852931191382, correct: 14
Epoch: 35/500, loss: 21.950919028226004, correct: 14
Epoch: 36/500, loss: 23.527206301240877, correct: 14
Epoch: 37/500, loss: 20.397915257683138, correct: 14
Epoch: 38/500, loss: 18.257330952541075, correct: 14
Epoch: 39/500, loss: 17.262814200969956, correct: 15
Epoch: 40/500, loss: 15.938997986100741, correct: 15
Epoch: 41/500, loss: 15.30450225139116, correct: 15
Epoch: 42/500, loss: 13.76082614807055, correct: 14
Epoch: 43/500, loss: 13.531381373629324, correct: 15
Epoch: 44/500, loss: 11.878248752881605, correct: 14
Epoch: 45/500, loss: 14.204585467592553, correct: 14
Epoch: 46/500, loss: 12.622949974653414, correct: 15
Epoch: 47/500, loss: 11.625500160769917, correct: 14
Epoch: 48/500, loss: 11.948373606019732, correct: 14
Epoch: 49/500, loss: 13.049873479178514, correct: 14
Epoch: 50/500, loss: 11.060214198800516, correct: 15
Epoch: 51/500, loss: 9.565396827749787, correct: 14
Epoch: 52/500, loss: 10.718108059514565, correct: 14
Epoch: 53/500, loss: 8.937390020394007, correct: 15
Epoch: 54/500, loss: 10.988843625654798, correct: 15
Epoch: 55/500, loss: 10.674110387536821, correct: 15
Epoch: 56/500, loss: 8.717849931874664, correct: 15
Epoch: 57/500, loss: 9.349152578526207, correct: 14
Epoch: 58/500, loss: 9.347195281185895, correct: 15
Epoch: 59/500, loss: 7.499532147922942, correct: 14
Epoch: 60/500, loss: 8.156249151044, correct: 15
Epoch: 61/500, loss: 9.69777969091778, correct: 15
Epoch: 62/500, loss: 19.043172988002695, correct: 7
Epoch: 63/500, loss: 41.27864602978719, correct: 13
Epoch: 64/500, loss: 32.42252765133993, correct: 14
Epoch: 65/500, loss: 23.97488841491034, correct: 14
Epoch: 66/500, loss: 20.361017033084735, correct: 14
Epoch: 67/500, loss: 22.180551183968856, correct: 14
Epoch: 68/500, loss: 21.99977838086566, correct: 14
Epoch: 69/500, loss: 19.77059170412999, correct: 14
Epoch: 70/500, loss: 17.224084128626558, correct: 14
Epoch: 71/500, loss: 19.426256223782286, correct: 14
Epoch: 72/500, loss: 16.687113868332748, correct: 14
Epoch: 73/500, loss: 15.869959342741037, correct: 14
Epoch: 74/500, loss: 17.195187748295496, correct: 14
Epoch: 75/500, loss: 20.311861059016184, correct: 13
Epoch: 76/500, loss: 15.000621184826556, correct: 14
Epoch: 77/500, loss: 13.918579008665722, correct: 14
Epoch: 78/500, loss: 13.144382120404773, correct: 15
Epoch: 79/500, loss: 10.856266862801426, correct: 14
Epoch: 80/500, loss: 10.412205097363548, correct: 14
Epoch: 81/500, loss: 9.344116494875138, correct: 14
Epoch: 82/500, loss: 9.869108889194479, correct: 14
Epoch: 83/500, loss: 8.428004518824253, correct: 14
Epoch: 84/500, loss: 7.124968341887057, correct: 14
Epoch: 85/500, loss: 6.776049821082005, correct: 14
Epoch: 86/500, loss: 7.324218866348716, correct: 15
Epoch: 87/500, loss: 7.376631397596716, correct: 14
Epoch: 88/500, loss: 9.156015456056462, correct: 14
Epoch: 89/500, loss: 8.952729555151766, correct: 14
Epoch: 90/500, loss: 8.248537886875386, correct: 14
Epoch: 91/500, loss: 9.508558903318294, correct: 14
Epoch: 92/500, loss: 7.016759877995212, correct: 14
Epoch: 93/500, loss: 6.490619388232751, correct: 14
Epoch: 94/500, loss: 7.276804807770762, correct: 14
Epoch: 95/500, loss: 6.915736231042278, correct: 14
Epoch: 96/500, loss: 6.166938435560386, correct: 14
Epoch: 97/500, loss: 6.619906410296658, correct: 14
Epoch: 98/500, loss: 5.657610929774892, correct: 14
Epoch: 99/500, loss: 6.304930942331192, correct: 14
Epoch: 100/500, loss: 6.8100895278230205, correct: 14
Epoch: 101/500, loss: 6.537977787549247, correct: 14
Epoch: 102/500, loss: 6.405327469718182, correct: 14
Epoch: 103/500, loss: 5.607471451369088, correct: 14
Epoch: 104/500, loss: 5.9588692857942265, correct: 15
Epoch: 105/500, loss: 5.635216006588564, correct: 15
Epoch: 106/500, loss: 4.048519578665528, correct: 15
Epoch: 107/500, loss: 3.7181615072110508, correct: 15
Epoch: 108/500, loss: 4.193893312485221, correct: 14
Epoch: 109/500, loss: 3.9171278757496584, correct: 15
Epoch: 110/500, loss: 5.03266916998148, correct: 14
Epoch: 111/500, loss: 4.171428564944035, correct: 14
Epoch: 112/500, loss: 3.0433458564254767, correct: 15
Epoch: 113/500, loss: 3.300623874067839, correct: 14
Epoch: 114/500, loss: 3.09671658646252, correct: 14
Epoch: 115/500, loss: 3.311176080631971, correct: 14
Epoch: 116/500, loss: 2.638514689116251, correct: 14
Epoch: 117/500, loss: 3.7156167430695985, correct: 14
Epoch: 118/500, loss: 3.095292767229327, correct: 14
Epoch: 119/500, loss: 3.2510916266216165, correct: 14
Epoch: 120/500, loss: 3.3848103307526096, correct: 14
Epoch: 121/500, loss: 2.8280856565777355, correct: 14
Epoch: 122/500, loss: 3.6158123141445446, correct: 14
Epoch: 123/500, loss: 3.450576445116324, correct: 15
Epoch: 124/500, loss: 3.0439921941720742, correct: 14
Epoch: 125/500, loss: 4.468977413527141, correct: 14
Epoch: 126/500, loss: 3.7933145342402264, correct: 14
Epoch: 127/500, loss: 4.0250351782088245, correct: 14
Epoch: 128/500, loss: 3.57188034114055, correct: 14
Epoch: 129/500, loss: 3.100787056730642, correct: 14
```

Sentiment Analysis

<img src="img/sentiment_error.png" width="500px">
<img src="img/sentiment_loss.png" width="500px">

```bash
Initializing model...
Epoch: 1/250, loss: 31.432134232978264, train accuracy: 0.5022222222222222
Epoch: 2/250, loss: 31.150649895067712, train accuracy: 0.5088888888888888
Epoch: 3/250, loss: 30.98989967613567, train accuracy: 0.5155555555555555
Epoch: 4/250, loss: 30.768281049426903, train accuracy: 0.5644444444444444
Epoch: 5/250, loss: 30.560310450416726, train accuracy: 0.5911111111111111
Epoch: 6/250, loss: 30.374481584907045, train accuracy: 0.5733333333333334
Epoch: 7/250, loss: 30.069003619960064, train accuracy: 0.6111111111111112
Epoch: 8/250, loss: 29.909112136542053, train accuracy: 0.6533333333333333
Epoch: 9/250, loss: 29.641550631180056, train accuracy: 0.6244444444444445
Epoch: 10/250, loss: 29.267335704082598, train accuracy: 0.6511111111111111
Epoch: 11/250, loss: 28.689760993536137, train accuracy: 0.6555555555555556
Epoch: 12/250, loss: 28.268316162893782, train accuracy: 0.6777777777777778
Epoch: 13/250, loss: 27.681309708892492, train accuracy: 0.7022222222222222
Epoch: 14/250, loss: 27.27011797088478, train accuracy: 0.7044444444444444
Epoch: 15/250, loss: 26.4051188572359, train accuracy: 0.7511111111111111
Epoch: 16/250, loss: 26.106164866811696, train accuracy: 0.74
Epoch: 17/250, loss: 25.533115659878487, train accuracy: 0.7244444444444444
Epoch: 18/250, loss: 24.880353059146067, train accuracy: 0.7266666666666667
Epoch: 19/250, loss: 24.194505185562196, train accuracy: 0.7666666666666667
Epoch: 20/250, loss: 23.14516066985068, train accuracy: 0.7755555555555556
Epoch: 21/250, loss: 22.42883551731623, train accuracy: 0.7888888888888889
Epoch: 22/250, loss: 21.831371820788213, train accuracy: 0.7844444444444445
Epoch: 23/250, loss: 21.31681324485781, train accuracy: 0.8088888888888889
Epoch: 24/250, loss: 21.847035498942102, train accuracy: 0.7577777777777778
Epoch: 25/250, loss: 20.889129838374647, train accuracy: 0.7977777777777778
Epoch: 26/250, loss: 20.459693965122327, train accuracy: 0.7755555555555556
Epoch: 27/250, loss: 19.772129436509953, train accuracy: 0.7933333333333333
Epoch: 28/250, loss: 19.161196882819635, train accuracy: 0.8066666666666666
Epoch: 29/250, loss: 18.11550462275991, train accuracy: 0.8088888888888889
Epoch: 30/250, loss: 17.337963458768098, train accuracy: 0.8511111111111112
Epoch: 31/250, loss: 17.70364123725329, train accuracy: 0.8088888888888889
Epoch: 32/250, loss: 17.168495903961404, train accuracy: 0.8222222222222222
Epoch: 33/250, loss: 15.613134703701295, train accuracy: 0.8333333333333334
Epoch: 34/250, loss: 16.492025370998352, train accuracy: 0.8288888888888889
Epoch: 35/250, loss: 16.59769875341387, train accuracy: 0.8244444444444444
Epoch: 36/250, loss: 15.048638168435875, train accuracy: 0.8555555555555555
Epoch: 37/250, loss: 15.017458509517434, train accuracy: 0.84
Epoch: 38/250, loss: 15.595098930334201, train accuracy: 0.8422222222222222
Epoch: 39/250, loss: 14.60104049340114, train accuracy: 0.8466666666666667
Epoch: 40/250, loss: 14.932377344829986, train accuracy: 0.8377777777777777
Epoch: 41/250, loss: 14.503628603078498, train accuracy: 0.8422222222222222
Epoch: 42/250, loss: 13.093403905686765, train accuracy: 0.8911111111111111
Epoch: 43/250, loss: 13.28797797367871, train accuracy: 0.8622222222222222
Epoch: 44/250, loss: 13.273949668047365, train accuracy: 0.8711111111111111
Epoch: 45/250, loss: 12.977892513885758, train accuracy: 0.8466666666666667
Epoch: 46/250, loss: 12.773346934207325, train accuracy: 0.8666666666666667
Epoch: 47/250, loss: 12.00043456783899, train accuracy: 0.88
Epoch: 48/250, loss: 11.837955126832165, train accuracy: 0.8755555555555555
Epoch: 49/250, loss: 12.599880323096679, train accuracy: 0.8711111111111111
Epoch: 50/250, loss: 11.60878046422687, train accuracy: 0.8888888888888888
Epoch: 51/250, loss: 12.345942987499846, train accuracy: 0.8577777777777778
Epoch: 52/250, loss: 12.120496114707427, train accuracy: 0.8644444444444445
Epoch: 53/250, loss: 12.400338628295948, train accuracy: 0.8288888888888889
Epoch: 54/250, loss: 12.448921468583318, train accuracy: 0.8466666666666667
Epoch: 55/250, loss: 11.680908608899884, train accuracy: 0.8488888888888889
Epoch: 56/250, loss: 11.230991721075506, train accuracy: 0.8644444444444445
Epoch: 57/250, loss: 11.239778519010294, train accuracy: 0.8555555555555555
Epoch: 58/250, loss: 10.546812313339823, train accuracy: 0.86
Epoch: 59/250, loss: 11.87257671817395, train accuracy: 0.8511111111111112
Epoch: 60/250, loss: 11.455089742776643, train accuracy: 0.8688888888888889
Epoch: 61/250, loss: 9.938012978779222, train accuracy: 0.8977777777777778
Epoch: 62/250, loss: 10.364312516475648, train accuracy: 0.8755555555555555
Epoch: 63/250, loss: 11.444679614147038, train accuracy: 0.8466666666666667
Epoch: 64/250, loss: 9.94735063854257, train accuracy: 0.8755555555555555
Epoch: 65/250, loss: 10.858444426164207, train accuracy: 0.88
Epoch: 66/250, loss: 10.809491523118542, train accuracy: 0.8888888888888888
Epoch: 67/250, loss: 8.845334388278786, train accuracy: 0.8933333333333333
Epoch: 68/250, loss: 10.14549599792885, train accuracy: 0.8577777777777778
Epoch: 69/250, loss: 10.562410412503125, train accuracy: 0.8644444444444445
Epoch: 70/250, loss: 10.552364417148057, train accuracy: 0.8422222222222222
Epoch: 71/250, loss: 10.589564464030829, train accuracy: 0.8644444444444445
Epoch: 72/250, loss: 9.828585583264134, train accuracy: 0.8622222222222222
Epoch: 73/250, loss: 9.871679523759902, train accuracy: 0.8622222222222222
Epoch: 74/250, loss: 10.15537498000789, train accuracy: 0.8688888888888889
Epoch: 75/250, loss: 8.53657522418968, train accuracy: 0.8955555555555555
Epoch: 76/250, loss: 9.600865853769987, train accuracy: 0.8911111111111111
Epoch: 77/250, loss: 10.124662406570504, train accuracy: 0.8488888888888889
Epoch: 78/250, loss: 10.996523269846191, train accuracy: 0.8533333333333334
Epoch: 79/250, loss: 8.205091470479806, train accuracy: 0.8844444444444445
Epoch: 80/250, loss: 8.800434057837364, train accuracy: 0.8844444444444445
Epoch: 81/250, loss: 8.542603643942973, train accuracy: 0.8822222222222222
Epoch: 82/250, loss: 9.92638689942078, train accuracy: 0.84
Epoch: 83/250, loss: 8.92349317175891, train accuracy: 0.8933333333333333
Epoch: 84/250, loss: 9.98794702192112, train accuracy: 0.8666666666666667
Epoch: 85/250, loss: 8.894921653797237, train accuracy: 0.8622222222222222
```