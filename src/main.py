from openpyxl import load_workbook
from CSR import *
import os.path


# getUsers returns a list of the (unique) users
def getUsers(ws):
    user_list = []

    user_column = ws['A']
    for cell in range(2, len(user_column)):
        if user_column[cell].value not in user_list:
            user_list.append(user_column[cell].value)

    return user_list


def main():
    wb = load_workbook(os.path.dirname(__file__) + '../Data_InCarMusic.xlsx')
    ws = wb["TrainSet"]

    user_list = getUsers(ws)

    driving_style_csr = CSR()
    landscape_csr = CSR()
    mood_csr = CSR()
    natural_csr = CSR()
    road_type_csr = CSR()
    sleepiness_csr = CSR()
    traffic_cond_csr = CSR()
    weather_csr = CSR()

    driving_style_csr.build_from_excel(ws, user_list, dimension=4)
    landscape_csr.build_from_excel(ws, user_list, dimension=5)
    mood_csr.build_from_excel(ws, user_list, dimension=6)
    natural_csr.build_from_excel(ws, user_list, dimension=7)
    road_type_csr.build_from_excel(ws, user_list, dimension=8)
    sleepiness_csr.build_from_excel(ws, user_list, dimension=9)
    traffic_cond_csr.build_from_excel(ws, user_list, dimension=10)
    weather_csr.build_from_excel(ws, user_list, dimension=11)

    ds_sim = driving_style_csr.calculate_cosine_sim()
    print(ds_sim.csr_dict['val'])

    # Redundantly generates sim matrix again, but nicely outputs data
    # in the following form: 'i j value'
    driving_style_csr.calculate_and_output_cosine_sim('sim.txt', -1)


main()