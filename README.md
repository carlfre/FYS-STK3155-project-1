# FYS-STK3155-project-1

--------------------------
### by: Carl Fredrik Nordbø Knutsen, Halvor Tyseng, Benedict Leander Skålevik Parton  
--------------------------
--------------------------
## Folder structure:
--------------------------
/notebooks:
  "Some problems explored with jupyter notebooks in an early stage of the project"
/plots
  "Different plots created thought the work on this project"
  /other_plots
    "Final plots used in report, note most of theese where created with snip_tools after plt.show() in franke_analysis.py and terraindata_analysis.py" 
  /plots_report
    "plots created when solving each problem seperatelly, not relevant for the report"
    /oppgave_d
    /oppgave_c
    /oppgave_e
    /oppgave_g
/src
  "Source code used to create, analyse and plot results"
  SRMT_Saarland.tif 
    note: datafile, should optimally be a /data folder
  analysis.py
  franke_analysis.py
  main.py
    note: Originally solving of the problems presented in project1 problem text. (Only deals with problems conserning the franke function)
          The problems are seperated in different functions, which can be called in function main()
  regression_tools.py
  terraindata_analysis.py
    note: The 3D-plots in the report is created here by calling the main() function with differnt parameters
  three_d_plotting.py
  
--------------------------
## Recreation of results
--------------------------
The results presented in the report is created with franke_analysis.py and terraindata_analysis.py. Theese utilize fuction and classes in the other .py files. The different analysis done in the report is seperated in different function wich can be called in main.
### NOTE: One have to change the main() fuction in both franke_analysis.py and terraindata_analysis.py to recreate each of plots and results we have used in the plot

