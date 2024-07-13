import moduls.model
import sys
from streamlit import cli as stcli



model = moduls.model.edit_model()

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", 'C:\Project\Corn\streamlit_app.py']
    sys.exit(stcli.main())
