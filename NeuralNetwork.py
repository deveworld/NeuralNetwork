import os
import sys
import numpy
import pickle
import imageio
import tempfile
import scipy.special
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Gui(QMainWindow):

	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.image = QImage(QSize(300, 300), QImage.Format_RGB32)
		self.image.fill(Qt.white)
		self.lr = 0.1
		self.atr = False
		self.teachnum = 0
		self.drawing = False
		self.brush_size = 10
		self.brush_color = Qt.black
		self.last_point = QPoint()

		with open('data\\model.dat', 'rb') as file:
			try:
				self.wih = pickle.load(file)
				self.who = pickle.load(file)
			except EOFError:
				self.wih = numpy.random.normal(0.0, pow(100, -0.5), (100, 784))
				self.who = numpy.random.normal(0.0, pow(10, -0.5), (10, 100))

		menubar = self.menuBar()
		menubar.setNativeMenuBar(False)
		menu = menubar.addMenu('Menu')

		load_model_action = QAction('모델 불러오기', self)
		load_model_action.setShortcut('Ctrl+L')
		load_model_action.triggered.connect(self.load_model)

		save_model_action = QAction('모델 저장하기', self)
		save_model_action.setShortcut('Ctrl+S')
		save_model_action.triggered.connect(self.save_model)

		clear_action = QAction('지우기', self)
		clear_action.setShortcut('Ctrl+C')
		clear_action.triggered.connect(self.clear)

		ateach_action = QAction('자동 가르치기', self)
		ateach_action.triggered.connect(self.autoteach)

		test_action = QAction('성능 테스트 하기', self)
		test_action.triggered.connect(self.autotest)

		teach_action = QAction('가르치기', self)
		teach_action.setShortcut('Ctrl+V')
		teach_action.triggered.connect(self.teach)

		setting_action = QAction('가르치기 설정', self)
		setting_action.setShortcut('Ctrl+X')
		setting_action.triggered.connect(self.setting)

		menu.addAction(load_model_action)
		menu.addAction(save_model_action)
		menu.addAction(clear_action)
		menu.addAction(ateach_action)
		menu.addAction(test_action)
		menu.addAction(teach_action)
		menu.addAction(setting_action)

		self.statusbar = self.statusBar()

		self.setWindowIcon(QIcon('data/ai.png'))
		self.setWindowTitle('NeuralNetwork')
		self.resize(300, 300)
		self.center()
		self.show()

	def setting(self):
		teachn, ok = QInputDialog.getInt(self, '입력창', '가르칠 숫자를 입력해주세요:')
		if ok:
			self.teachnum = teachn
			QMessageBox.question(self, '안내창', '설정이 저장 되었습니다.', QMessageBox.Yes)
		else:
			QMessageBox.question(self, '안내창', '취소되었습니다.', QMessageBox.Yes)
			return

	def clear(self):
		self.image.fill(Qt.white)
		self.update()
		self.statusbar.clearMessage()

	def load_model(self):
		QMessageBox.question(self, '안내창', '모델이 있는 폴더를 지정해주세요.', QMessageBox.Yes)
		path = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), './', QFileDialog.ShowDirsOnly)
		if os.path.exists(path+'\\'+'model.dat'):
			with open(path+'\\'+'model.dat', 'rb') as file:
				try:
					self.wih = pickle.load(file)
					self.who = pickle.load(file)
				except EOFError:
					QMessageBox.question(self, '안내창', '모델 파일이 손상 되었습니다.', QMessageBox.Yes)
					return
				else:
					QMessageBox.question(self, '안내창', '모델을 성공적으로 불러왔습니다!', QMessageBox.Yes)
		else:
			QMessageBox.question(self, '안내창', '해당 폴더에 모델이 없습니다.', QMessageBox.Yes)

	def save_model(self):
		QMessageBox.question(self, '안내창', '모델을 저장 할 폴더를 지정해주세요.', QMessageBox.Yes)
		path = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), './', QFileDialog.ShowDirsOnly)
		if os.path.exists(path+'\\'+'model.dat'):
			QMessageBox.question(self, '안내창', '이미 해당 폴더에 모델이 존재합니다.', QMessageBox.Yes)
			return
		else:
			with open(path+'\\'+'model.dat', 'wb') as file:
				pickle.dump(self.wih, file)
				pickle.dump(self.who, file)
			QMessageBox.question(self, '안내창', '성공적으로 저장되었습니다!', QMessageBox.Yes)

	def getimg(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			self.image.scaled(28, 28).save(tmpdir+'\\'+'a.png')
			img_array = imageio.imread(tmpdir+'\\'+'a.png', as_gray=True)

		img_data  = 255.0 - img_array.reshape(784)
		img_data = (img_data / 255.0 * 0.99) + 0.01

		return img_data

	def mouseReleaseEvent(self, e):
		if e.button() == Qt.LeftButton:
			self.drawing = False

			if self.atr == False:
				img = self.getimg()
				n = nNq(self, self.wih, self.who, img)
				n.res.connect(self.result)
				n.start()
			else:
				self.statusbar.showMessage('자동으로 가르치는 중 입니다. 기다려주세요.')

	def teach(self):
		img = self.getimg()
		targets = numpy.zeros(10) + 0.01
		targets[int(self.teachnum)] = 0.99
		n = nNt(self, self.wih, self.who, self.lr, img, targets)
		n.rwih.connect(self.setwih)
		n.rwho.connect(self.setwho)
		n.start()

	def setwih(self, value):
		self.wih = value

	def setwho(self, value):
		self.who = value

	def result(self, value):
		self.statusbar.showMessage('숫자 ' + str(value) + '입니다.')

	def paintEvent(self, event):
		canvas = QPainter(self)
		canvas.drawImage(self.rect(), self.image, self.image.rect())

	def mousePressEvent(self, event):
		if event.button() == Qt.LeftButton:
			self.drawing = True
			self.last_point = event.pos()

	def mouseMoveEvent(self, event):
		if (event.buttons() & Qt.LeftButton) & self.drawing:
			painter = QPainter(self.image)
			painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
			painter.drawLine(self.last_point, event.pos())
			self.last_point = event.pos()
			self.update()

	def autoteachdone(self):
		self.atr = False
		QMessageBox.question(self, '안내창', '자동으로 가르치기가 종료되었습니다!', QMessageBox.Yes)

	def autoteach(self):
		QMessageBox.question(self, '안내창', '자동으로 가르치기가 시작되었습니다!', QMessageBox.Yes)

		t = nNatr(self, self.wih, self.who, self.lr)
		t.done.connect(self.autoteachdone)
		self.atr = True
		t.start()

	def autotest(self):
		QMessageBox.question(self, '안내창', '성능 테스트가 시작되었습니다!', QMessageBox.Yes)

		t = nNate(self, self.wih, self.who)
		t.done.connect(self.autotestdone)
		t.start()

	def autotestdone(self, value):
		QMessageBox.question(self, '안내창', '성능 테스트가 종료되었습니다!, 성능 : '+str(value), QMessageBox.Yes)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

class nNate(QThread):

	done = pyqtSignal(float)

	def __init__(self, parent, wih, who):
		super().__init__(parent)
		self.parent = parent

		self.wih = wih
		self.who = who

		self.activation_function = lambda x: scipy.special.expit(x)

	def run(self):
		test_data_file = open("data/mnist_test.csv", 'r')
		test_data_list = test_data_file.readlines()
		test_data_file.close()

		scorecard = []

		for record in test_data_list:
			all_values = record.split(',')
		
			correct_label = int(all_values[0])
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			qoutputs = self.query(inputs)
			label = numpy.argmax(qoutputs)
			if (label == correct_label):
				scorecard.append(1)
			else:
				scorecard.append(0)

		scorecard_array = numpy.asfarray(scorecard)
		self.done.emit(scorecard_array.sum() / scorecard_array.size)

	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs =self.activation_function(final_inputs)

		return final_outputs

class nNatr(QThread):

	done = pyqtSignal()

	def __init__(self, parent, wih, who, learningrate):
		super().__init__(parent)
		self.parent = parent

		self.wih = wih
		self.who = who

		self.lr = learningrate

		self.activation_function = lambda x: scipy.special.expit(x)

	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs =self.activation_function(final_inputs)

		outputs_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, outputs_errors)

		self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		
	def run(self):
		training_data_file = open("data/mnist_train.csv", 'r')
		training_data_list = training_data_file.readlines()
		training_data_file.close()

		for record in training_data_list:
			all_values = record.split(',')
	
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			targets = numpy.zeros(10) + 0.01
			targets[int(all_values[0])] = 0.99

			self.train(inputs, targets)
		self.done.emit()

class nNq(QThread):

	res = pyqtSignal(str)

	def __init__(self, parent, wih, who, imglist):
		super().__init__(parent)
		self.parent = parent

		self.wih = wih
		self.who = who

		self.imglist = imglist

		self.activation_function = lambda x: scipy.special.expit(x)

	def run(self):
		inputs = numpy.array(self.imglist, ndmin=2).T
		
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs =self.activation_function(final_inputs)
		print(final_outputs)
		print(numpy.argmax(final_outputs))
		self.res.emit(str(numpy.argmax(final_outputs)))

class nNt(QThread):

	rwho = pyqtSignal(numpy.ndarray)
	rwih = pyqtSignal(numpy.ndarray)

	def __init__(self, parent, wih, who, learning_rate, inputs_list, targets_list):
		super().__init__(parent)
		self.parent = parent

		self.wih = wih
		self.who = who

		self.lr = learning_rate

		self.inputs_list = inputs_list
		self.targets_list = targets_list

		self.activation_function = lambda x: scipy.special.expit(x)

	def run(self):
		inputs = numpy.array(self.inputs_list, ndmin=2).T
		targets = numpy.array(self.targets_list, ndmin=2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs =self.activation_function(final_inputs)

		outputs_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, outputs_errors)

		self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

		self.rwho.emit(self.who)
		self.rwih.emit(self.wih)
		
if __name__ == '__main__':

	try:
		os.chdir(sys._MEIPASS)
		print(sys._MEIPASS)
	except:
		os.chdir(os.getcwd())

	app = QApplication(sys.argv)
	ex = Gui()
	sys.exit(app.exec_())