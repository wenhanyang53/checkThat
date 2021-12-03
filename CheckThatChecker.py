from tkinter import *
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def build_data_set():
    # checking if enough input is provided
    not_enough_info = False
    if not (var_feature_n.get() or var_feature_l.get() or var_feature_e.get() or var_feature_c.get()
            or var_feature_w.get()):
        messagebox.showerror("No feature group!", "You have not selected a feature group!")
        not_enough_info = True
    if not (var_train_1.get() or var_train_2.get() or var_train_3.get()
            or (len(text_train_x.get("1.0", 'end-1c')) != 0) & (len(text_train_y.get("1.0", 'end-1c')) != 0)):
        messagebox.showerror("No training dataset!", "You have not selected a training dataset!")
        not_enough_info = True
    if not (var_test_1.get() or var_test_2.get() or var_test_3.get() or var_test_4.get() or var_test_5.get()
            or var_test_6.get() or var_test_7.get()
            or (len(text_test_x.get("1.0", 'end-1c')) != 0) & (len(text_test_y.get("1.0", 'end-1c')) != 0)):
        messagebox.showerror("No test dataset!", "You have not selected a test dataset!")
        not_enough_info = True
    if not_enough_info:
        return

    # Building the complete training dataset
    list_train_features = []
    list_train_labels = []
    if var_train_1.get():
        list_train_features.append('train-NLECW-features-file1.csv')
        list_train_labels.append('train-labels-file1.csv')
    if var_train_2.get():
        list_train_features.append('train-NLECW-features-file2.csv')
        list_train_labels.append('train-labels-file2.csv')
    if var_train_3.get():
        list_train_features.append('train-NLECW-features-file3.csv')
        list_train_labels.append('train-labels-file3.csv')
    if (len(text_train_x.get("1.0", 'end-1c')) != 0) & (len(text_train_y.get("1.0", 'end-1c')) != 0):
        list_train_features.append(text_train_x.get("1.0", 'end-1c'))
        list_train_labels.append(text_train_y.get("1.0", 'end-1c'))
    x_train = pd.concat(map(pd.read_csv, list_train_features))
    y_train = pd.concat(map(pd.read_csv, list_train_labels))['Label']

    # building the complete testing dataset
    list_test_features = []
    list_test_labels = []
    if var_test_1.get():
        list_test_features.append('test-NLECW-features-task1-en-file1.csv')
        list_test_labels.append('test-labels-task1-en-file1.csv')
    if var_test_2.get():
        list_test_features.append('test-NLECW-features-task1-en-file2.csv')
        list_test_labels.append('test-labels-task1-en-file2.csv')
    if var_test_3.get():
        list_test_features.append('test-NLECW-features-task1-en-file3.csv')
        list_test_labels.append('test-labels-task1-en-file3.csv')
    if var_test_4.get():
        list_test_features.append('test-NLECW-features-task1-en-file4.csv')
        list_test_labels.append('test-labels-task1-en-file4.csv')
    if var_test_5.get():
        list_test_features.append('test-NLECW-features-task1-en-file5.csv')
        list_test_labels.append('test-labels-task1-en-file5.csv')
    if var_test_6.get():
        list_test_features.append('test-NLECW-features-task1-en-file6.csv')
        list_test_labels.append('test-labels-task1-en-file6.csv')
    if var_test_7.get():
        list_test_features.append('test-NLECW-features-task1-en-file7.csv')
        list_test_labels.append('test-labels-task1-en-file7.csv')
    if (len(text_test_x.get("1.0", 'end-1c')) != 0) & (len(text_test_y.get("1.0", 'end-1c')) != 0):
        list_test_features.append(text_test_x.get("1.0", 'end-1c'))
        list_test_labels.append(text_test_y.get("1.0", 'end-1c'))
    x_test = pd.concat(map(pd.read_csv, list_test_features))
    y_test = pd.concat(map(pd.read_csv, list_test_labels))['Label']

    # modifying dataset based on selected labels
    included_features_train = []
    included_features_test = []
    if var_feature_n.get():
        included_features_train.append(x_train.loc[:, 'N1':'N5'])
        included_features_test.append(x_test.loc[:, 'N1':'N5'])
    if var_feature_l.get():
        included_features_train.append(x_train.loc[:, 'L1':'L136'])
        included_features_test.append(x_test.loc[:, 'L1':'L136'])
    if var_feature_e.get():
        included_features_train.append(x_train.loc[:, 'E1':'E4'])
        included_features_test.append(x_test.loc[:, 'E1':'E4'])
    if var_feature_c.get():
        included_features_train.append(x_train.loc[:, 'C1':'C338'])
        included_features_test.append(x_test.loc[:, 'C1':'C338'])
    if var_feature_w.get():
        included_features_train.append(x_train.loc[:, 'W1':'W300'])
        included_features_test.append(x_test.loc[:, 'W1':'W300'])
    x_train_final = pd.concat(included_features_train, axis=1)
    x_test_final = pd.concat(included_features_test, axis=1)

    return {'x_train': x_train_final, 'x_test': x_test_final, 'y_train': y_train, 'y_test': y_test}


# Prints results, including a heatmap in a new window
def results(y_pred, data_sets):

    text_result.insert('1.0', classification_report(data_sets['y_test'], y_pred))

    # New window for the resulting confusion matrix
    result_window = Tk()
    result_window.title("Prediction Heatmap")

    # Creates new figure
    fig = plt.figure(figsize=(10, 16))

    # Calculates a confusion matrix
    cm = confusion_matrix(data_sets['y_test'], y_pred)
    cm_sum = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    cm_1 = cm[0][0]/cm_sum*100
    cm_2 = cm[0][1]/cm_sum*100
    cm_3 = cm[1][0]/cm_sum*100
    cm_4 = cm[1][1]/cm_sum*100
    cm_per = [[cm_1, cm_2], [cm_3, cm_4]]
    ax = plt.subplot()

    # Draws the figure in the designated window
    canvas = FigureCanvasTkAgg(fig, result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Draws the heatmap using the calculated confusion matrix
    sb.set(font_scale=1.5)
    ax = sb.heatmap(cm_per, vmin=0, vmax=100, annot=True, fmt = '.2f', annot_kws={"fontsize":15})
    for t in ax.texts: t.set_text(t.get_text() + " %")

    # Labels the heatmap for readability
    ax.set_xlabel('Predicted values',fontsize=20)
    ax.set_ylabel('True values',fontsize=20)
    ax.set_title('Confusion Matrix')

    result_window.mainloop()


def k_neighbors():
    print('Running K-Neighbors with input values!')
    # retrieving the combined datasets
    data_sets = build_data_set()

    # return if datasets weren't returned successfully
    if data_sets is None:
        return

    # running the predictions
    knn = KNeighborsClassifier(n_neighbors=1).fit(data_sets['x_train'], data_sets['y_train'])
    y_pred = knn.predict(data_sets['x_test'])
    results(y_pred, data_sets)


def naive_bayes():
    print('Running Naive Bayes with input values!')
    # retrieving the combined datasets
    data_sets = build_data_set()

    # return if datasets weren't returned successfully
    if data_sets is None:
        return

    # running the predictions
    gnb = GaussianNB().fit(data_sets['x_train'], data_sets['y_train'])
    y_pred = gnb.predict(data_sets['x_test'])
    results(y_pred, data_sets)


def linearSVC():
    print('Running Linear Support Vector Classification with input values!')
    # retrieving the combined datasets
    data_sets = build_data_set()

    # return if datasets weren't returned successfully
    if data_sets is None:
        return
    svc = LinearSVC().fit(data_sets['x_train'], data_sets['y_train'])
    y_pred = svc.predict(data_sets['x_test'])
    results(y_pred, data_sets)


def decision_tree():
    print('Running a Decision Tree with input values!')
    # retrieving the combined datasets
    data_sets = build_data_set()

    # return if datasets weren't returned successfully
    if data_sets is None:
        return
    tree = DecisionTreeClassifier(random_state=0).fit(data_sets['x_train'], data_sets['y_train'])
    y_pred = tree.predict(data_sets['x_test'])
    results(y_pred, data_sets)

# initialize UI window
window = Tk()
window.title("AnalyzeLab")

# feature selection group
group_features = LabelFrame(window, text="Choose Features", font= 'Calibri 12 bold', padx=5, pady=5)
group_features.grid(row=0, column=0, padx=5, pady=5, rowspan=2, sticky = W)
var_feature_n = BooleanVar()
var_feature_l = BooleanVar()
var_feature_e = BooleanVar()
var_feature_c = BooleanVar()
var_feature_w = BooleanVar()
cb_feature_n = Checkbutton(group_features, text="Nutritional label based features", font= 'Calibri 12', variable=var_feature_n)
cb_feature_n.pack(side=TOP, anchor=W)
cb_feature_l = Checkbutton(group_features, text="Linguistic features", font= 'Calibri 12', variable=var_feature_l)
cb_feature_l.pack(side=TOP, anchor=W)
cb_feature_e = Checkbutton(group_features, text="Entity features", font= 'Calibri 12', variable=var_feature_e)
cb_feature_e.pack(side=TOP, anchor=W)
cb_feature_c = Checkbutton(group_features, text="Category features", font= 'Calibri 12', variable=var_feature_c)
cb_feature_c.pack(side=TOP, anchor=W)
cb_feature_w = Checkbutton(group_features, text="Word-embedding based features", font= 'Calibri 12', variable=var_feature_w)
cb_feature_w.pack(side=TOP, anchor=W)

# training dataset selection group
group_train = LabelFrame(window, text="Choose Training Datasets", font='Calibri 12 bold', height=130, width=320, padx=5, pady=5)
group_train.pack_propagate(False)
group_train.grid(row=0, column=2, padx=5, pady=5)
var_train_1 = BooleanVar()
var_train_2 = BooleanVar()
var_train_3 = BooleanVar()
cb_train_1 = Checkbutton(group_train, text="1st presidential", font= 'Calibri 12', variable=var_train_1)
cb_train_1.pack(side=TOP, anchor=W)
cb_train_2 = Checkbutton(group_train, text="2nd presidential", font= 'Calibri 12', variable=var_train_2)
cb_train_2.pack(side=TOP, anchor=W)
cb_train_3 = Checkbutton(group_train, text="Vice-Presidential", font= 'Calibri 12', variable=var_train_3)
cb_train_3.pack(side=TOP, anchor=W)

# custom training dataset group
group_train_text = LabelFrame(window, text='Names of custom training datasets '
                                           '\n(include .csv ending; 1. features, 2. labels)', font= 'Calibri 12')
group_train_text.grid(row=8, column=2, padx=5, pady=5)
text_train_x = Text(group_train_text, height=1, width=45)
text_train_x.pack(side=TOP, anchor=W)
text_train_y = Text(group_train_text, height=1, width=45)
text_train_y.pack(side=TOP, anchor=W)

# testing dataset selection group
group_test = LabelFrame(window, text="Choose Test Datasets", font= 'Calibri 12 bold', height=240, width=400, padx=5, pady=5)
group_test.pack_propagate(False)
group_test.grid(row=0, column=3, sticky=W, padx=5, pady=5)
var_test_1 = BooleanVar()
var_test_2 = BooleanVar()
var_test_3 = BooleanVar()
var_test_4 = BooleanVar()
var_test_5 = BooleanVar()
var_test_6 = BooleanVar()
var_test_7 = BooleanVar()
cb_test_1 = Checkbutton(group_test, text="3rd Presidential", font= 'Calibri 12', variable=var_test_1)
cb_test_1.pack(side=TOP, anchor=W)
cb_test_2 = Checkbutton(group_test, text="9th Democratic", font= 'Calibri 12', variable=var_test_2)
cb_test_2.pack(side=TOP, anchor=W)
cb_test_3 = Checkbutton(group_test, text="Donald Trump Acceptance", font= 'Calibri 12', variable=var_test_3)
cb_test_3.pack(side=TOP, anchor=W)
cb_test_4 = Checkbutton(group_test, text="Donald Trump at World Economic Forum", font= 'Calibri 12', variable=var_test_4)
cb_test_4.pack(side=TOP, anchor=W)
cb_test_5 = Checkbutton(group_test, text="Donald Trump at Tax Reform Event", font= 'Calibri 12', variable=var_test_5)
cb_test_5.pack(side=TOP, anchor=W)
cb_test_6 = Checkbutton(group_test, text="Donald Trump's Address to Congress", font= 'Calibri 12', variable=var_test_6)
cb_test_6.pack(side=TOP, anchor=W)
cb_test_7 = Checkbutton(group_test, text="Donald Trump's Miami Speech", font= 'Calibri 12', variable=var_test_7)
cb_test_7.pack(side=TOP, anchor=W)

# custom testing dataset group
group_test_text = LabelFrame(window, text='Names of custom testing datasets ' 
                                          '\n(include .csv ending; 1. features, 2. labels)', font= 'Calibri 12')
group_test_text.grid(row=8, column=3, padx=5, pady=5, sticky = W)
text_test_x = Text(group_test_text, height=1, width=45)
text_test_x.pack(side=TOP, anchor=W)
text_test_y = Text(group_test_text, height=1, width=45)
text_test_y.pack(side=TOP, anchor=W)

# buttons for running different algorithms
button_k_neighbors = Button(window, text="Run K-Neighbors!", font= 'Calibri 12', command=k_neighbors)
button_k_neighbors.grid(row=14, column=0,  pady=5)
button_naive_bayes = Button(window, text="Run Naive Bayes!", font= 'Calibri 12', command=naive_bayes)
button_naive_bayes.grid(row=14, column=1,  pady=10)
button_svc = Button(window, text="Run Linear SVC!", font= 'Calibri 12', command=linearSVC)
button_svc.grid(row=14, column=2, pady=10)
button_tree = Button(window, text="Run a Decision Tree!", font= 'Calibri 12', command=decision_tree)
button_tree.grid(row=14, column=3, pady=10)

# textbox for printed result
text_result = Text(window, height=10, width=60)
text_result.grid(row=11, column=2, padx=10)

# running the actual window / application
window.mainloop()
