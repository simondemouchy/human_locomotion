# RAMP starting kit on the Human Locomotion data set

[![Build Status](https://travis-ci.com/ramp-kits/human-locomotion.svg?branch=master)](https://travis-ci.com/ramp-kits/human-locomotion)

Associated publication: Truong, C., Barrois-Müller, R., Moreau, T., Provost, C., Vienne-Jumeau, A., Moreau, A., Vidal, P.-P., Vayatis, N., Buffat, S., Yelnik, A., Ricard, D., & Oudre, L. (2019). A data set for the study of human locomotion with inertial measurements units. Image Processing On Line (IPOL), 9. [[doi]](https://doi.org/10.5201/ipol.2019.265) [[pdf]](http://deepcharles.github.io/files/ipol-walk-data-2019.pdf) [[online demo]](http://ipolcore.ipol.im/demo/clientApp/demo.html?id=265)

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with a [dedicated notebook](https://github.com/ramp-kits/human-locomotion/blob/master/human_locomotion_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
