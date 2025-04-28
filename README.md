# FMambaIR
Code of the paper "FMambaIR: A Hybrid State Space Model and Frequency Domain for Image Restoration"
# Datasets
 ```sh
      |--- datasets
             |--- {datasets_name}
                   |--- train
                         |--- input
                         |--- GT
                   |--- test
                         |--- iuput
                         |--- GT
                   
   ```
### Test

   Test the FMambaIR model:
   
   ```sh
   python test.py 
   ```
   
   The pretrained model is saved at ./checkpoints/UIEBD/*.pth.
