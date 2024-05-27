import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QTextEdit, 
    QScrollArea, QFrame, QCheckBox, QLayoutItem
)
from PyQt5.QtCore import Qt
from functools import partial

# Sample data for multiple annotations
annotations = [
    {
        "chat": [
            {"role": "system", "content": "Welcome to the chat! How can I assist you today?"},
            {"role": "user", "content": "I'm looking for information on your services."},
            {"role": "system", "content": "We offer a range of services including A, B, and C. Which one are you interested in?"},
            {"role": "user", "content": "I'm interested in service B."},
        ],
        "criteria": ["Relevance", "Politeness", "Clarity"],
        "binary_criteria": ["Appropriateness"]
    },
    {
        "chat": [
            {"role": "system", "content": "How can I help you today?"},
            {"role": "user", "content": "Can you tell me about your pricing?"},
            {"role": "system", "content": "Sure, we have different pricing plans for different services. Which service are you interested in?"},
            {"role": "user", "content": "I'm interested in the premium plan."},
        ],
        "criteria": ["Relevance", "Politeness", "Clarity"],
        "binary_criteria": ["Appropriateness"]
    }
]

class ChatAnnotationInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.annotation_index = 0
        self.annotations = annotations
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Chat Annotation Interface')
        self.setGeometry(100, 100, 800, 600)
        
        self.layout = QVBoxLayout()
        self.chat_layout = QVBoxLayout()
        self.criteria_layout = QVBoxLayout()
        
        # Create chat display
        self.chat_label = QLabel('Chat')
        self.chat_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.chat_layout.addWidget(self.chat_label)
        
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_widget = QWidget()
        self.chat_area.setWidget(self.chat_widget)
        self.chat_messages_layout = QVBoxLayout(self.chat_widget)
        
        self.chat_layout.addWidget(self.chat_area)
        
        # Create criteria display
        self.criteria_label = QLabel('Criteria')
        self.criteria_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.criteria_layout.addWidget(self.criteria_label)
        
        self.layout.addLayout(self.chat_layout)
        self.layout.addLayout(self.criteria_layout)
        
        # Set stretch factors for the main layout
        self.layout.setStretch(0, 3)  # Chat layout takes 3 parts of vertical space
        self.layout.setStretch(1, 1)  # Criteria layout takes 1 part of vertical space
        
        self.setLayout(self.layout)
        
        # Load the first annotation
        self.load_annotation(self.annotation_index)
    
    def load_annotation(self, index):
        annotation = self.annotations[index]
        
        # Clear previous chat messages
        for i in reversed(range(self.chat_messages_layout.count())):
            widget = self.chat_messages_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        # Display chat messages
        for entry in annotation["chat"]:
            role = entry["role"].capitalize()
            content = entry["content"]
            
            message_layout = QHBoxLayout()
            role_label = QLabel(role)
            role_label.setStyleSheet("font-weight: bold;")
            
            separator = QFrame()
            separator.setFrameShape(QFrame.VLine)
            separator.setFrameShadow(QFrame.Sunken)
            
            content_label = QTextEdit()
            content_label.setReadOnly(True)
            content_label.setPlainText(content)
            content_label.setStyleSheet("background-color: #f0f0f0;" if entry["role"] == "system" else "background-color: #d0f0d0;")
            
            message_layout.addWidget(role_label)
            message_layout.addWidget(separator)
            message_layout.addWidget(content_label)
            
            # Set stretch factors
            message_layout.setStretch(0, 1)  # Role label takes 1 part of space
            message_layout.setStretch(1, 0)  # Separator takes minimal space
            message_layout.setStretch(2, 4)  # Content label takes 4 parts of space
            
            self.chat_messages_layout.addLayout(message_layout)
        
        # Clear previous criteria
        while self.criteria_layout.count():
            item: QLayoutItem = self.criteria_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
        self.ratings = {}
        self.binary_ratings = {}
        self.criteria_labels = []
        self.widgets = []  # To manage focus and tab order
        
        # Display criteria
        for i, criterion in enumerate(annotation["criteria"]):
            criterion_layout = QHBoxLayout()
            
            label = QLabel(criterion)
            criterion_layout.addWidget(label)
            self.criteria_labels.append(label)
            
            rating_slider = QSlider(Qt.Horizontal)
            rating_slider.setMinimum(1)
            rating_slider.setMaximum(5)
            rating_slider.setValue(1)
            rating_slider.setObjectName(f"slider_{i}")
            rating_slider.installEventFilter(self)  # Install event filter
            rating_slider.setFocusPolicy(Qt.StrongFocus)
            criterion_layout.addWidget(rating_slider)
            
            rating_value_label = QLabel("1")
            criterion_layout.addWidget(rating_value_label)
            
            rating_slider.valueChanged.connect(partial(self.update_slider_label, rating_value_label))
            
            self.criteria_layout.addLayout(criterion_layout)
            self.ratings[criterion] = rating_slider
            self.widgets.append(rating_slider)
        
        # Display binary criteria
        for i, criterion in enumerate(annotation["binary_criteria"]):
            criterion_layout = QHBoxLayout()
            
            label = QLabel(criterion)
            criterion_layout.addWidget(label)
            self.criteria_labels.append(label)
            
            rating_checkbox = QCheckBox()
            rating_checkbox.setObjectName(f"checkbox_{i}")
            rating_checkbox.installEventFilter(self)  # Install event filter
            rating_checkbox.setFocusPolicy(Qt.StrongFocus)
            criterion_layout.addWidget(rating_checkbox)
            
            self.criteria_layout.addLayout(criterion_layout)
            self.binary_ratings[criterion] = rating_checkbox
            self.widgets.append(rating_checkbox)
        
        # Add submit button
        self.submit_button = QPushButton('Submit Ratings')
        self.submit_button.clicked.connect(self.submit_ratings)
        self.submit_button.setFocusPolicy(Qt.NoFocus)  # Prevent focus on the button
        self.criteria_layout.addWidget(self.submit_button)
        self.widgets.append(self.submit_button)
        
        # Set initial focus
        self.current_focus_index = 0
        self.setFocusToWidget(self.current_focus_index)
    
    def update_slider_label(self, label, value):
        label.setText(str(value))

    def eventFilter(self, source, event):
        if source in self.widgets:
            if event.type() == event.FocusIn:
                self.updateCriterionLabel(self.widgets.index(source), True)
            elif event.type() == event.FocusOut:
                self.updateCriterionLabel(self.widgets.index(source), False)
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        key = event.key()
        current_widget = self.widgets[self.current_focus_index]
        print(f"Key Pressed: {key}, Current Widget: {current_widget.objectName()}")
        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5]:
            if isinstance(current_widget, QSlider):
                current_widget.setValue(key - Qt.Key_0)
        elif key == Qt.Key_0:
            if isinstance(current_widget, QCheckBox):
                current_widget.setChecked(False)
        elif key == Qt.Key_1:
            if isinstance(current_widget, QCheckBox):
                current_widget.setChecked(True)
        elif key == Qt.Key_Tab:
            self.move_focus_forward()
        elif key == Qt.Key_Backtab:
            self.move_focus_backward()
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self.submit_ratings()
    
    def move_focus_forward(self):
        self.current_focus_index = (self.current_focus_index + 1) % len(self.widgets)
        self.setFocusToWidget(self.current_focus_index)
    
    def move_focus_backward(self):
        self.current_focus_index = (self.current_focus_index - 1) % len(self.widgets)
        self.setFocusToWidget(self.current_focus_index)
    
    def setFocusToWidget(self, index):
        print(f"Setting focus to widget: {self.widgets[index].objectName()}")
        self.widgets[index].setFocus()

    def updateCriterionLabel(self, index, focused):
        if index < len(self.criteria_labels):
            label = self.criteria_labels[index]
            label.setStyleSheet("font-weight: bold;" if focused else "")

    def submit_ratings(self):
        rating_values = {criterion: slider.value() for criterion, slider in self.ratings.items()}
        binary_rating_values = {criterion: checkbox.isChecked() for criterion, checkbox in self.binary_ratings.items()}
        print("Submitted Ratings:", rating_values)
        print("Submitted Binary Ratings:", binary_rating_values)
        
        self.annotation_index += 1
        if self.annotation_index < len(self.annotations):
            self.load_annotation(self.annotation_index)
        else:
            self.restart_application()

    def restart_application(self):
        print("Restarting application...")
        QApplication.quit()
        QCoreApplication.exit(0)
        self.__init__()
        self.show()

def main():
    app = QApplication(sys.argv)
    ex = ChatAnnotationInterface()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

