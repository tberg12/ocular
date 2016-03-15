package edu.berkeley.cs.nlp.ocular.main.gui;

import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TrainLanguageModelGUI {

//	private int HEIGHT = 400;
//
//	private JFrame frame;
//	private JPanel panel_labels;
//	private JPanel panel_inputs;
//
//	private JLabel label_title1;
//	private JLabel label_title2;
//
//	private JLabel label_outputLmPath;
//	private JLabel label_inputTextPath;
//	private JLabel label_languagePriors;
//	private JLabel label_pKeepSameLanguage;
//	private JLabel label_alternateSpellingReplacementPaths;
//	private JLabel label_insertLongS;
//	private JLabel label_removeDiacritics;
//	private JLabel label_explicitCharacterSet;
//	private JLabel label_charN;
//	private JLabel label_lmPower;
//	private JLabel label_lmCharCount;
//	private JLabel label_go;
//
//	private JTextField input_outputLmPath;
//	private JTextField input_inputTextPath;
//	private JTextField input_languagePriors;
//	private JTextField input_pKeepSameLanguage;
//	private JTextField input_alternateSpellingReplacementPaths;
//	private JCheckBox input_insertLongS;
//	private JCheckBox input_removeDiacritics;
//	private JTextField input_explicitCharacterSet;
//	private JTextField input_lmPower;
//	private JTextField input_charN;
//	private JTextField input_lmCharCount;
//	private JButton input_go;
//
//	/**
//	 * Launch the application.
//	 */
//	public static void main(String[] args) {
//		EventQueue.invokeLater(new Runnable() {
//			public void run() {
//				try {
//					TrainLanguageModelGUI window = new TrainLanguageModelGUI();
//					window.frame.setVisible(true);
//				}
//				catch (Exception e) {
//					e.printStackTrace();
//				}
//			}
//		});
//	}
//
//	/**
//	 * Create the application.
//	 */
//	public TrainLanguageModelGUI() {
//		initialize();
//	}
//
//	/**
//	 * Initialize the contents of the frame.
//	 */
//	private void initialize() {
//		frame = new JFrame();
//		frame.setBounds(100, 100, 1000, HEIGHT);
//		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//		// frame.getContentPane().setLayout(new GridLayout(1, 2, 0, 0));
//		frame.getContentPane().setLayout(new GridLayout2(1, 2, 1, 1));
//		// frame.getContentPane().setLayout(new GridBagLayout());
//
//		panel_labels = new JPanel();
//		frame.getContentPane().add(panel_labels);
//		panel_labels.setLayout(new GridLayout(0, 1, 0, 0));
//		panel_labels.setPreferredSize(new Dimension(40, HEIGHT));
//
//		label_title1 = new JLabel("  Ocular");
//		label_title1.setFont(new Font("Lucida Grande", Font.BOLD, 18));
//		panel_labels.add(label_title1);
//
//		label_outputLmPath = new JLabel("LM path ");
//		label_outputLmPath.setToolTipText("Output Language Model file path. Required.");
//		label_outputLmPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_outputLmPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_outputLmPath);
//		label_inputTextPath = new JLabel("Input text path ");
//		label_inputTextPath.setToolTipText("Path to the text files (or directory hierarchies) for training the LM. For each entry, the entire directory will be recursively searched for any files that do not start with .. For a multilingual (code-switching) model, give multiple comma-separated files with language names: \"english->texts/english/,spanish->texts/spanish/,french->texts/french/\". If spaces are used, be sure to wrap the whole string with \"quotes\".). Required.");
//		label_inputTextPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_inputTextPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_inputTextPath);
//		label_languagePriors = new JLabel("Language priors ");
//		label_languagePriors.setToolTipText("Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language, prior pairs: english->0.7,spanish->0.2,french->0.1. If spaces are used, be sure to wrap the whole string with \"quotes\". (Only relevant if multiple languages used.) Default: uniform priors");
//		label_languagePriors.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_languagePriors);
//		label_pKeepSameLanguage = new JLabel("pKeepSameLanguage ");
//		label_pKeepSameLanguage.setToolTipText("Prior probability of sticking with the same language when moving between words in a code-switch model transition model. (Only relevant if multiple languages used.) Default: 0.999999");
//		label_pKeepSameLanguage.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_pKeepSameLanguage);
//		label_alternateSpellingReplacementPaths = new JLabel("Alternate spelling replacement paths ");
//		label_alternateSpellingReplacementPaths.setToolTipText("Paths to Alternate Spelling Replacement files. Give multiple comma-separated language, path pairs: english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt. If spaces are used, be sure to wrap the whole string with \"quotes\". Any languages for which no replacements are needed can be safely ignored. Default: no replacements");
//		label_alternateSpellingReplacementPaths.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_alternateSpellingReplacementPaths);
//		label_insertLongS = new JLabel("Insert long-S ");
//		label_insertLongS.setToolTipText("Use separate character type for long s. Default: false");
//		label_insertLongS.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_insertLongS);
//		label_removeDiacritics = new JLabel("Remove diacritics? ");
//		label_removeDiacritics.setToolTipText("Remove diacritics? Default: false");
//		label_removeDiacritics.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_removeDiacritics);
//		label_explicitCharacterSet = new JLabel("Explicit character set ");
//		label_explicitCharacterSet.setToolTipText("A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters. Not currently implemented. Default: ...");
//		label_explicitCharacterSet.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_explicitCharacterSet);
//		label_charN = new JLabel("N-gram 'n' ");
//		label_charN.setToolTipText("LM character n-gram length. Default: 6");
//		label_charN.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_charN);
//		label_lmPower = new JLabel("lmPower ");
//		label_lmPower.setToolTipText("Exponent on LM scores.");
//		label_lmPower.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_lmPower);
//		label_lmCharCount = new JLabel("LM character count ");
//		label_lmCharCount.setToolTipText("Number of characters to use for training the LM. Use -1 to indicate that the full training data should be used. Default: -1");
//		label_lmCharCount.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_lmCharCount);
//		label_go = new JLabel("");
//		label_go.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_go);
//
//		panel_inputs = new JPanel();
//		frame.getContentPane().add(panel_inputs);
//		panel_inputs.setLayout(new GridLayout(0, 1, 0, 0));
//
//		label_title2 = new JLabel();
//		panel_inputs.add(label_title2);
//
//		input_outputLmPath = new JTextField();
//		input_outputLmPath.setToolTipText("Output Language Model file path. Required.");
//		panel_inputs.add(input_outputLmPath);
//		input_inputTextPath = new JTextField();
//		input_inputTextPath.setToolTipText("Path to the text files (or directory hierarchies) for training the LM. For each entry, the entire directory will be recursively searched for any files that do not start with .. For a multilingual (code-switching) model, give multiple comma-separated files with language names: \"english->texts/english/,spanish->texts/spanish/,french->texts/french/\". If spaces are used, be sure to wrap the whole string with \"quotes\".). Required.");
//		panel_inputs.add(input_inputTextPath);
//		input_languagePriors = new JTextField();
//		input_languagePriors.setToolTipText("Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language, prior pairs: english->0.7,spanish->0.2,french->0.1. If spaces are used, be sure to wrap the whole string with \"quotes\". (Only relevant if multiple languages used.) Default: uniform priors");
//		panel_inputs.add(input_languagePriors);
//		input_pKeepSameLanguage = new JTextField();
//		input_pKeepSameLanguage.setToolTipText("Prior probability of sticking with the same language when moving between words in a code-switch model transition model. (Only relevant if multiple languages used.) Default: 0.999999");
//		input_pKeepSameLanguage.setText("0.999999");
//		panel_inputs.add(input_pKeepSameLanguage);
//		input_alternateSpellingReplacementPaths = new JTextField();
//		input_alternateSpellingReplacementPaths.setToolTipText("Paths to Alternate Spelling Replacement files. Give multiple comma-separated language, path pairs: english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt. If spaces are used, be sure to wrap the whole string with \"quotes\". Any languages for which no replacements are needed can be safely ignored. Default: no replacements");
//		panel_inputs.add(input_alternateSpellingReplacementPaths);
//		input_insertLongS = new JCheckBox();
//		input_insertLongS.setToolTipText("Use separate character type for long s. Default: false");
//		panel_inputs.add(input_insertLongS);
//		input_removeDiacritics = new JCheckBox();
//		input_removeDiacritics.setToolTipText("Remove diacritics? Default: false");
//		panel_inputs.add(input_removeDiacritics);
//		input_explicitCharacterSet = new JTextField();
//		input_explicitCharacterSet.setToolTipText("A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters. Not currently implemented. Default: ...");
//		panel_inputs.add(input_explicitCharacterSet);
//		TranscribeOrTrainFontGUI.setEnabled(input_explicitCharacterSet, false);
//		input_charN = new JTextField();
//		input_charN.setToolTipText("LM character n-gram length. Default: 6");
//		input_charN.setText("6");
//		panel_inputs.add(input_charN);
//		input_lmPower = new JTextField();
//		input_lmPower.setToolTipText("Exponent on LM scores.");
//		input_lmPower.setText("4.0");
//		panel_inputs.add(input_lmPower);
//		input_lmCharCount = new JTextField();
//		input_lmCharCount.setToolTipText("Number of characters to use for training the LM. Use -1 to indicate that the full training data should be used. Default: -1");
//		input_lmCharCount.setText("-1");
//		panel_inputs.add(input_lmCharCount);
//		input_go = new JButton();
//		input_go.setMaximumSize(new Dimension(20, 100));
//		input_go.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent e) {
//				InitializeLanguageModel exec = new InitializeLanguageModel();
//				exec.outputLmPath = input_outputLmPath.getText();
//				exec.inputTextPath = input_inputTextPath.getText();
//				exec.languagePriors = input_languagePriors.getText();
//				exec.pKeepSameLanguage = Double.valueOf(input_pKeepSameLanguage.getText());
//				exec.alternateSpellingReplacementPaths = input_alternateSpellingReplacementPaths.getText();
//				exec.insertLongS = input_insertLongS.isSelected();
//				exec.removeDiacritics = input_removeDiacritics.isSelected();
//				exec.explicitCharacterSet = null; // input_explicitCharacterSet.getText();
//				exec.charN = Integer.valueOf(input_charN.getText());
//				exec.lmPower = Double.valueOf(input_lmPower.getText());
//				exec.lmCharCount = Integer.valueOf(input_lmCharCount.getText());
//				exec.run();
//			}
//		});
//		input_go.setText("Train LM");
//		panel_inputs.add(input_go);
//
//	}

}
