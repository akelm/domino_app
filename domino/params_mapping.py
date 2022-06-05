from PyQt5.QtWidgets import QLineEdit, QCheckBox, QComboBox, QButtonGroup, QPushButton, QSpinBox, QDoubleSpinBox

getters = dict(((QLineEdit, lambda x: float(QLineEdit.text(x).replace(',', '.')) if QLineEdit.text(x) else -1),
                (QCheckBox, QCheckBox.isChecked), (QComboBox, QComboBox.currentIndex),
                (QButtonGroup, lambda x: -QButtonGroup.checkedId(x) - 2), (QPushButton, QPushButton.text),
                (QSpinBox, QSpinBox.value), (QDoubleSpinBox, QDoubleSpinBox.value),))
