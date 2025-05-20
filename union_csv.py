"""Este programa es para unir csvs. Para crear una nueva unión, definir la función y llamarla en la parte de abajo
"""

import pandas as pd


def unir_portugal_trad():
    df_p_0_99 = pd.read_csv("portugal_trad_0_99.csv")
    df_p_100_199 = pd.read_csv("portugal_trad_100_199.csv")
    df_p_200_299 = pd.read_csv("portugal_trad_200_299.csv")
    df_p_300_399 = pd.read_csv("portugal_trad_300_399.csv")
    df_p_400_555 = pd.read_csv("portugal_trad_400_555.csv")

    df_portugal_trad = pd.concat([df_p_0_99,df_p_100_199,df_p_200_299,df_p_300_399,df_p_400_555],ignore_index=True)
    df_portugal_trad.to_csv("portugal_trad.csv",index=False)

def unir_spain_trad():
    df_s_0_99 = pd.read_csv("spain_trad_0_99.csv")
    df_s_100_199 = pd.read_csv("spain_trad_100_199.csv")
    df_s_200_299 = pd.read_csv("spain_trad_200_299.csv")
    df_s_300_399 = pd.read_csv("spain_trad_300_399.csv")
    df_s_400_499 = pd.read_csv("spain_trad_400_499.csv")
    df_s_500_633 = pd.read_csv("spain_trad_500_633.csv")

    df_spain_trad = pd.concat([df_s_0_99,df_s_100_199,df_s_200_299,df_s_300_399,df_s_400_499,df_s_500_633], ignore_index=True)
    df_spain_trad.to_csv("spain_trad.csv", index=False)

def unir_portugal_spain_trad_scores():
    df_p = pd.read_csv("portugal_trad_scores.csv")
    df_s = pd.read_csv("spain_trad_scores.csv")

    df_p_s = pd.concat([df_p,df_s], ignore_index=True)
    df_p_s.to_csv("portugal_spain_trad_scores.csv", index=False)

#función a ejecutar:
#unir_portugal_trad()
#unir_spain_trad()
#unir_portugal_spain_trad_scores()


