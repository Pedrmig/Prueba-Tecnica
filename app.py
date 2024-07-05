import streamlit as st
import joblib
import numpy as np

train_model = joblib.load('modelo_train_prueba.pkl')

def main():
    st.title("Formulario de Datos")

    # Mapeamento de Género
    genero_map = {"Masculino": 1, "Femenino": 0}
    genero_selecionado = st.selectbox("Género", list(genero_map.keys()))
    CODE_GENDER = genero_map[genero_selecionado]

    car_map = {"No": 0, "Sí": 1}
    car_selecionado = st.selectbox("Posee Coche", list(car_map.keys()))
    FLAG_OWN_CAR = car_map[car_selecionado]
    
    Own_realty_map = {"No": 0, "Sí": 1}
    Own_realty_selecionado = st.selectbox("Posee Inmueble", list(Own_realty_map.keys()))
    FLAG_OWN_REALTY = Own_realty_map[Own_realty_selecionado]
    
    CNT_CHILDREN = st.number_input("Número de Hijos", min_value=0, step=1)
    
    AMT_INCOME_TOTAL = st.number_input("Ingreso Total Anual", step=1.0)
    
    income_type_map = {"Commercial associate": 0,"Pensioner": 1, "No Informado": 2,"State servant": 3,"Working": 4}
    income_type_selecionado = st.selectbox("Tipo de Ingreso", list(income_type_map.keys()))
    NAME_INCOME_TYPE = income_type_map[income_type_selecionado]
    
    education_map = {"Academic degree":0, "Higher education": 1,"Incomplete higher": 2, "Lower secondary": 3,"Secondary / secondary special": 4, "Other": 5}
    education_selecionado = st.selectbox("Nivel de Educación", list(education_map.keys()))
    NAME_EDUCATION_TYPE = education_map[education_selecionado]
    
    family_status_map = {"Civil marriage": 0, "Married": 1,"Separated": 2,"Single / not married": 3, "Unknown": 4}
    family_status_selecionado = st.selectbox("Estado Civil", list(family_status_map.keys()))
    NAME_FAMILY_STATUS = family_status_map[family_status_selecionado]
    
    housing_type_map = {"House / apartment": 0, "Rented apartment": 1, "With parents": 2, "Municipal apartment": 3, "Office apartment": 4, "Co-op apartment": 5}
    housing_type_selecionado = st.selectbox("Tipo de Vivienda", list(housing_type_map.keys()))
    NAME_HOUSING_TYPE = housing_type_map[housing_type_selecionado]
    
    DAYS_EMPLOYED = st.number_input("Días Empleado", step=1.0)
    
    mobil_map = {"No": 0, "Sí": 1}
    mobil_selecionado = st.selectbox("Posee Teléfono Móvil", list(mobil_map.keys()))
    FLAG_MOBIL = mobil_map[mobil_selecionado]
    
    work_phone_map = {"No": 0, "Sí": 1}
    work_phone_selecionado = st.selectbox("Posee Teléfono de Trabajo", list(work_phone_map.keys()))
    FLAG_WORK_PHONE = work_phone_map[work_phone_selecionado]
    
    phone_map = {"No": 0, "Sí": 1}
    phone_selecionado = st.selectbox("Posee Teléfono en casa", list(phone_map.keys()))
    FLAG_PHONE = phone_map[phone_selecionado]
    
    email_map = {"No": 0, "Sí": 1}
    email_selecionado = st.selectbox("Posee Email", list(email_map.keys()))
    FLAG_EMAIL = email_map[email_selecionado]
    
    occupation_map = {'other':18, 'Medicine staff':11, 'Sales staff':14, 'Security staff':16,'Core staff':3, 'Drivers':4, 'Laborers':8, 'Accountants':0,
       'Cooking staff':2, 'Cleaning staff':1, 'Managers':10,'High skill tech staff':6, 'Low-skill Laborers':9,'Private service staff':12, 'Secretaries':15, 'Waiters/barmen staff':17,'HR staff':5, 'Realty agents':13, 'IT staff':7}
    occupation_selecionado = st.selectbox("Tipo de Ocupación", list(occupation_map.keys()))
    OCCUPATION_TYPE = occupation_map[occupation_selecionado]
    
    CNT_FAM_MEMBERS = st.number_input("Número de Miembros en la Familia", min_value=1.0, step=1.0)
    
    ANO = st.number_input("Año de nacimiento", min_value=1800, max_value=2100, step=1)
    
    state_flag = {'CT':6, 'AR':2, 'DC':7, 'MD':13, 'FL':8, 'AL':1, 'CO':5, 'CA':4, 'AZ':3, 'KY':11, 'TN':17,'MA':12, 'GA':9, 'AK':0, 'VT':19, 'OK':14, 'II':10, 'US':18, 'PI':15, 'WB':20, 'RD':16}
    state_selecionado = st.selectbox("Estado que vives", list(state_flag.keys()))
    STATE = state_flag[state_selecionado]
   
    def predict(model, input_data):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    
    if st.button("Enviar"):
        datos = [
            CODE_GENDER,
            FLAG_OWN_CAR,
            FLAG_OWN_REALTY,
            CNT_CHILDREN,
            AMT_INCOME_TOTAL,
            NAME_INCOME_TYPE,
            NAME_EDUCATION_TYPE,
            NAME_FAMILY_STATUS,
            NAME_HOUSING_TYPE,
            DAYS_EMPLOYED,
            FLAG_MOBIL,
            FLAG_WORK_PHONE,
            FLAG_PHONE,
            FLAG_EMAIL,
            OCCUPATION_TYPE,
            CNT_FAM_MEMBERS,
            ANO,
            STATE
        ]
        
        predicted = predict(train_model, datos)
        st.write(f"El Cliente es considerado: {predicted:.2f}")

if __name__ == "__main__":
    main()
