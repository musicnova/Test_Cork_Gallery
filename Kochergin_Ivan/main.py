import moduls.model
import sys
from streamlit import __main__ as stcli

model = moduls.model.edit_model()


if __name__ == '__main__':
    sys.argv = ["streamlit", "run", 'C:\Project\Corn\streamlit_app.py', '--theme.base', 'dark', '--theme.primaryColor', '#F59A07', '--theme.font', 'serif']
    sys.exit(stcli.main())

