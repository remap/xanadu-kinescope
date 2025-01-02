


### Installation

```
cd fb_sender_example/
python -m venv venv
source venv/bin/activate
pip install requests google.auth
```

Obtain the secrets file and place in the directory.

### Run

```
python fb_sender_example.py
```

Which yields something like: 

```

% python fb_sender_example.py     

Refresh Firebase Access Token:
	Token: ya29.c.c0ASRK0Gbhg_-c5ZvO...
	Expiry: 2025-01-02 23:21:20.978987-08:00
{'energy': 22.921361662076208, 'accuracy': 75.91024432657449, 'lag': 90.92482487607185}
Firebase async return {'accuracy': 75.91024432657449, 'energy': 22.921361662076208, 'lag': 90.92482487607185}
{'energy': -43.57581958450276, 'accuracy': -12.473100676542884, 'lag': 44.61140569885916}
Firebase async return {'accuracy': -12.473100676542884, 'energy': -43.57581958450276, 'lag': 44.61140569885916}
{'energy': -3.563347726147015, 'accuracy': -19.347821885125654, 'lag': 21.0092979319179}
Firebase async return {'accuracy': -19.347821885125654, 'energy': -3.563347726147015, 'lag': 21.0092979319179}
{'energy': 59.10532848408718, 'accuracy': -10.003105944574457, 'lag': -58.78455021838968}
Firebase async return {'accuracy': -10.003105944574457, 'energy': 59.10532848408718, 'lag': -58.78455021838968}
{'energy': -66.27376404046086, 'accuracy': -62.24313259540209, 'lag': 18.680808795917955}
Firebase async return {'accuracy': -62.24313259540209, 'energy': -66.27376404046086, 'lag': 18.680808795917955}
{'energy': -10.608515246571784, 'accuracy': -16.95272653532365, 'lag': -20.596055078321314}
Firebase async return {'accuracy': -16.95272653532365, 'energy': -10.608515246571784, 'lag': -20.596055078321314}
{'energy': -2.074706563115214, 'accuracy': -68.89242428668751, 'lag': -83.96833918791606}
Firebase async return {'accuracy': -68.89242428668751, 'energy': -2.074706563115214, 'lag': -83.96833918791606}
{'energy': 58.76319329274844, 'accuracy': 96.69062631341419, 'lag': -10.975284101362305}
Firebase async return {'accuracy': 96.69062631341419, 'energy': 58.76319329274844, 'lag': -10.975284101362305}
{'energy': 28.344564338497378, 'accuracy': -63.892752110676824, 'lag': 2.95005704322611}
Firebase async return {'accuracy': -63.892752110676824, 'energy': 28.344564338497378, 'lag': 2.95005704322611}
```
