package edu.berkeley.cs.nlp.ocular.main.gui;

import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import edu.berkeley.cs.nlp.ocular.main.InitializeFont;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class InitializeFontGUI {

	private int HEIGHT = 250;

	private JFrame frame;
	private JPanel panel_labels;
	private JPanel panel_inputs;

	private JLabel label_title1;
	private JLabel label_title2;

	private JLabel label_inputLmPath;
	private JLabel label_outputFontPath;
	private JLabel label_numFontInitThreads;
	private JLabel label_templateMaxWidthFraction;
	private JLabel label_templateMinWidthFraction;
	private JLabel label_spaceMaxWidthFraction;
	private JLabel label_spaceMinWidthFraction;
	private JLabel label_go;

	private JTextField input_inputLmPath;
	private JTextField input_outputFontPath;
	private JTextField input_numFontInitThreads;
	private JTextField input_templateMaxWidthFraction;
	private JTextField input_templateMinWidthFraction;
	private JTextField input_spaceMaxWidthFraction;
	private JTextField input_spaceMinWidthFraction;
	private JButton input_go;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					InitializeFontGUI window = new InitializeFontGUI();
					window.frame.setVisible(true);
				}
				catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public InitializeFontGUI() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 1000, HEIGHT);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		// frame.getContentPane().setLayout(new GridLayout(1, 2, 0, 0));
		frame.getContentPane().setLayout(new GridLayout2(1, 2, 1, 1));
		// frame.getContentPane().setLayout(new GridBagLayout());

		panel_labels = new JPanel();
		frame.getContentPane().add(panel_labels);
		panel_labels.setLayout(new GridLayout(0, 1, 0, 0));
		panel_labels.setPreferredSize(new Dimension(30, HEIGHT));

		label_title1 = new JLabel("  Ocular");
		label_title1.setFont(new Font("Lucida Grande", Font.BOLD, 18));
		panel_labels.add(label_title1);

		label_inputLmPath = new JLabel("LM path ");
		label_inputLmPath.setToolTipText("Path to the language model file (so that it knows which characters to create images for). Required.");
		label_inputLmPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
		label_inputLmPath.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_inputLmPath);
		label_outputFontPath = new JLabel("Font path ");
		label_outputFontPath.setToolTipText("Output font file path. Required.");
		label_outputFontPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
		label_outputFontPath.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_outputFontPath);
		label_numFontInitThreads = new JLabel("Number of threads ");
		label_numFontInitThreads.setToolTipText("Number of threads to use. Deafult: 8");
		label_numFontInitThreads.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_numFontInitThreads);
		label_templateMaxWidthFraction = new JLabel("templateMaxWidthFraction ");
		label_templateMaxWidthFraction.setToolTipText("Max template width as fraction of text line height. Default: 1.0");
		label_templateMaxWidthFraction.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_templateMaxWidthFraction);
		label_templateMinWidthFraction = new JLabel("templateMinWidthFraction ");
		label_templateMinWidthFraction.setToolTipText("Min template width as fraction of text line height. Default: 0.0");
		label_templateMinWidthFraction.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_templateMinWidthFraction);
		label_spaceMaxWidthFraction = new JLabel("spaceMaxWidthFraction ");
		label_spaceMaxWidthFraction.setToolTipText("Max space template width as fraction of text line height. Default: 1.0");
		label_spaceMaxWidthFraction.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_spaceMaxWidthFraction);
		label_spaceMinWidthFraction = new JLabel("spaceMinWidthFraction ");
		label_spaceMinWidthFraction.setToolTipText("Min space template width as fraction of text line height. Default: 0.0");
		label_spaceMinWidthFraction.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_spaceMinWidthFraction);
		label_go = new JLabel("");
		label_go.setHorizontalAlignment(SwingConstants.TRAILING);
		panel_labels.add(label_go);

		panel_inputs = new JPanel();
		frame.getContentPane().add(panel_inputs);
		panel_inputs.setLayout(new GridLayout(0, 1, 0, 0));

		label_title2 = new JLabel();
		panel_inputs.add(label_title2);

		input_inputLmPath = new JTextField();
		input_inputLmPath.setToolTipText("Path to the language model file (so that it knows which characters to create images for). Required.");
		panel_inputs.add(input_inputLmPath);
		input_outputFontPath = new JTextField();
		input_outputFontPath.setToolTipText("Output font file path. Required.");
		panel_inputs.add(input_outputFontPath);
		input_numFontInitThreads = new JTextField();
		input_numFontInitThreads.setText("8");
		input_numFontInitThreads.setToolTipText("Number of threads to use. Deafult: 8");
		panel_inputs.add(input_numFontInitThreads);
		input_templateMaxWidthFraction = new JTextField();
		input_templateMaxWidthFraction.setToolTipText("Max template width as fraction of text line height. Default: 1.0");
		input_templateMaxWidthFraction.setText("1.0");
		panel_inputs.add(input_templateMaxWidthFraction);
		input_templateMinWidthFraction = new JTextField();
		input_templateMinWidthFraction.setText("0.0");
		input_templateMinWidthFraction.setToolTipText("Min template width as fraction of text line height. Default: 0.0");
		panel_inputs.add(input_templateMinWidthFraction);
		input_spaceMaxWidthFraction = new JTextField();
		input_spaceMaxWidthFraction.setText("1.0");
		input_spaceMaxWidthFraction.setToolTipText("Max space template width as fraction of text line height. Default: 1.0");
		panel_inputs.add(input_spaceMaxWidthFraction);
		input_spaceMinWidthFraction = new JTextField();
		input_spaceMinWidthFraction.setText("0.0");
		input_spaceMinWidthFraction.setToolTipText("Min space template width as fraction of text line height. Default: 0.0");
		panel_inputs.add(input_spaceMinWidthFraction);
		input_go = new JButton();
		input_go.setMaximumSize(new Dimension(20, 100));
		input_go.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				InitializeFont exec = new InitializeFont();
				exec.inputLmPath = input_inputLmPath.getText();
				exec.outputFontPath = input_outputFontPath.getText();
				exec.numFontInitThreads = Integer.valueOf(input_numFontInitThreads.getText());
				exec.templateMaxWidthFraction = Double.valueOf(input_templateMaxWidthFraction.getText());
				exec.templateMinWidthFraction = Double.valueOf(input_templateMinWidthFraction.getText());
				exec.spaceMaxWidthFraction = Double.valueOf(input_spaceMaxWidthFraction.getText());
				exec.spaceMinWidthFraction = Double.valueOf(input_spaceMinWidthFraction.getText());
				exec.run(new ArrayList());
			}
		});
		input_go.setText("Train LM");
		panel_inputs.add(input_go);

	}

}
