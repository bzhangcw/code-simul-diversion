# Prepare the dataset from raw data

This step is optional because `./simulation/felony-1989-final.xlsx` is already provided.

Some intermediate files are provided under `./simulation/datasets/`; these are not needed in the code but good for reference.

A useful (verbose) dataset with some explanation to the columns are provided in `./simulation/datasets/felony-1989-with-explanation.xlsx`.

Unfortunately, we do not have an automated procedure for processing these datafiles. Below, we give some guidelines.

## Collecting the data for ICPSR

You will need raw data from ICPSR. Apply for access and download the source files from:

- [ICPSR 9574: Recidivism of Felons on Probation, 1986–1989](https://www.icpsr.umich.edu/web/NACJD/studies/9574) 
    - This is the probation dataset. Put it under `./simulation/datasets/ICPSR-9574/`
    - The databook is located at
    ```bash
    simulation/datasets/ICPSR-9574/DS0001/09574-0001-Codebook.txt
    ```
    It is quite difficult to read.  You can refer to a cleaned excel workbook in `./simulation/datasets/felony-1989-with-explanation.xlsx`.
- [ICPSR 9251: County and City Data Book, 1988](https://www.icpsr.umich.edu/web/ICPSR/studies/9251) 
    - This is the data for community information of each county or city (identified by FIPS code). Put it under `./simulation/datasets/ICPSR-9251/`
    - This dataset has three formats: ASCII, SAS, and SPSS. The contents are the same.

