package edu.berkeley.cs.nlp.ocular.main.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import edu.berkeley.cs.nlp.ocular.main.TrainFont;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TranscribeOrTrainFontGUI {

//	public static Color GRAY = new Color(238, 238, 238);
//
//	private int HEIGHT = 800;
//
//	private JFrame frame;
//	private JPanel panel_labels;
//	private JPanel panel_inputs;
//
//	private JLabel label_title1;
//	private JLabel label_title2;
//
//	private JLabel label_inputDocPath;
//	private JLabel label_numDocs;
//	private JLabel label_inputLmPath;
//	private JLabel label_inputFontPath;
//	private JLabel label_trainFont;
//	private JLabel label_numEMIters;
//	private JLabel label_outputPath;
//	private JLabel label_extractedLinesPath;
//	private JLabel label_outputFontPath;
//	private JLabel label_outputLmPath;
//	private JLabel label_allowLanguageSwitchOnPunct;
//	private JLabel label_binarizeThreshold;
//	private JLabel label_crop;
//	private JLabel label_uniformLineHeight;
//	private JLabel label_markovVerticalOffset;
//	private JLabel label_beamSize;
//	private JLabel label_emissionEngine;
//	private JLabel label_emissionEngine_2;
//	private JLabel label_emissionEngine_3;
//	private JLabel label_cudaDeviceID;
//	private JLabel label_numMstepThreads;
//	private JLabel label_numEmissionCacheThreads;
//	private JLabel label_numDecodeThreads;
//	private JLabel label_decodeBatchSize;
//	private JLabel label_paddingMinWidth;
//	private JLabel label_paddingMaxWidth;
//	private JLabel label_go;
//
//	private JTextField input_inputDocPath;
//	private JTextField input_numDocs;
//	private JTextField input_inputLmPath;
//	private JTextField input_inputFontPath;
//	private JCheckBox input_trainFont;
//	private JTextField input_numEMIters;
//	private JTextField input_outputPath;
//	private JTextField input_extractedLinesPath;
//	private JTextField input_outputFontPath;
//	private JTextField input_outputLmPath;
//	private JCheckBox input_allowLanguageSwitchOnPunct;
//	private JTextField input_binarizeThreshold;
//	private JCheckBox input_crop;
//	private JCheckBox input_uniformLineHeight;
//	private JCheckBox input_markovVerticalOffset;
//	private JTextField input_beamSize;
//	private ButtonGroup input_emissionEngine;
//	private JRadioButton input_emissionEngine_default;
//	private JRadioButton input_emissionEngine_opencl;
//	private JRadioButton input_emissionEngine_cuda;
//	private JTextField input_cudaDeviceID;
//	private JTextField input_numMstepThreads;
//	private JTextField input_numEmissionCacheThreads;
//	private JTextField input_numDecodeThreads;
//	private JTextField input_decodeBatchSize;
//	private JTextField input_paddingMinWidth;
//	private JTextField input_paddingMaxWidth;
//	private JButton input_go;
//
//	/**
//	 * Launch the application.
//	 */
//	public static void main(String[] args) {
//		EventQueue.invokeLater(new Runnable() {
//			public void run() {
//				try {
//					TranscribeOrTrainFontGUI window = new TranscribeOrTrainFontGUI();
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
//	public TranscribeOrTrainFontGUI() {
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
//		panel_labels.setPreferredSize(new Dimension(50, HEIGHT));
//
//		label_title1 = new JLabel("  Ocular");
//		label_title1.setFont(new Font("Lucida Grande", Font.BOLD, 18));
//		panel_labels.add(label_title1);
//
//		// names.foreach(n =>
//		// println(f"""label_$n%-30s = new JLabel(""); panel_1.add(label_$n);"""))
//		label_inputDocPath = new JLabel("Input path ");
//		label_inputDocPath.setToolTipText("Path of the directory that contains the input document images or pdfs. The entire directory will be recursively searched for any files that do not end in .txt (and that do not start with .). Required.");
//		label_inputDocPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_inputDocPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_inputDocPath);
//		label_numDocs = new JLabel("Maximum number of documents ");
//		label_numDocs.setToolTipText("Number of documents to use. Ignore or use -1 to use all documents. Default: -1");
//		label_numDocs.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_numDocs);
//		label_inputLmPath = new JLabel("LM path ");
//		label_inputLmPath.setToolTipText("Path to the language model file. Required.");
//		label_inputLmPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_inputLmPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_inputLmPath);
//		label_inputFontPath = new JLabel("Initial font path ");
//		label_inputFontPath.setToolTipText("Path of the font initializer file. Required.");
//		label_inputFontPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_inputFontPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_inputFontPath);
//		label_trainFont = new JLabel("Learn font? ");
//		label_trainFont.setToolTipText("Whether to learn the font from the input documents and write the font to a file. Default: false");
//		label_trainFont.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_trainFont);
//		label_numEMIters = new JLabel("Number of EM iterations ");
//		label_numEMIters.setToolTipText("Number of iterations of EM to use for font learning. Default: 3");
//		label_numEMIters.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_numEMIters);
//		label_outputPath = new JLabel("Output path ");
//		label_outputPath.setToolTipText("Path of the directory that will contain output transcriptions and line extractions. Required.");
//		label_outputPath.setFont(new Font("Lucida Grande", Font.BOLD, 13));
//		label_outputPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_outputPath);
//		label_extractedLinesPath = new JLabel("Extracted lines path ");
//		label_extractedLinesPath.setToolTipText("Path of the directory where the line-extraction images should be read/written. If the line files exist here, they will be used; if not, they will be extracted and then written here. Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document. If ignored, the document will simply be read from the original document image file, and no line images will be written.\nDefault: null (Don't read or write line image files.)");
//		label_extractedLinesPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_extractedLinesPath);
//		label_outputFontPath = new JLabel("Output font path ");
//		label_outputFontPath.setToolTipText("Path to write the learned font file to. (Only if trainFont is set to true.) Required if trainFont=true, otherwise ignored.");
//		label_outputFontPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_outputFontPath);
//		label_outputLmPath = new JLabel("Output LM path ");
//		label_outputLmPath.setToolTipText("Path to write the learned language model file to. (Only if trainFont is set to true.) Default: null (Don't write out the trained LM.)");
//		label_outputLmPath.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_outputLmPath);
//		label_allowLanguageSwitchOnPunct = new JLabel("Allow language switch on punctuation? ");
//		label_allowLanguageSwitchOnPunct.setToolTipText("A language model to be used to assign diacritics to the transcription output. Default: true");
//		label_allowLanguageSwitchOnPunct.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_allowLanguageSwitchOnPunct);
//		label_binarizeThreshold = new JLabel("Binarization threshold ");
//		label_binarizeThreshold.setToolTipText("Quantile to use for pixel value thresholding. (High values mean more black pixels.) Default: 0.12");
//		label_binarizeThreshold.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_binarizeThreshold);
//		label_crop = new JLabel("Crop? ");
//		label_crop.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_crop);
//		label_uniformLineHeight = new JLabel("Uniform line height? ");
//		label_uniformLineHeight.setToolTipText("Scale all lines to have the same height? Default: true");
//		label_uniformLineHeight.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_uniformLineHeight);
//		label_markovVerticalOffset = new JLabel("Markov vertical offset? ");
//		label_markovVerticalOffset.setToolTipText("Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.) Default: false");
//		label_markovVerticalOffset.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_markovVerticalOffset);
//		label_beamSize = new JLabel("Beam size? ");
//		label_beamSize.setToolTipText("Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.) Default: 10");
//		label_beamSize.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_beamSize);
//		label_emissionEngine = new JLabel("Emission engine ");
//		label_emissionEngine.setToolTipText("Engine to use for inner loop of emission cache computation. DEFAULT: Uses Java on CPU, which works on any machine but is the slowest method. OPENCL: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. CUDA: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation. Default: DEFAULT");
//		label_emissionEngine.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_emissionEngine);
//		label_emissionEngine_2 = new JLabel("");
//		panel_labels.add(label_emissionEngine_2);
//		label_emissionEngine_3 = new JLabel("");
//		panel_labels.add(label_emissionEngine_3);
//		label_cudaDeviceID = new JLabel("CUDA device ID ");
//		label_cudaDeviceID.setToolTipText("GPU ID when using CUDA emission engine. Default: 0\n\n");
//		label_cudaDeviceID.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_cudaDeviceID);
//		label_numMstepThreads = new JLabel("Number of M-step threads ");
//		label_numMstepThreads.setToolTipText("Number of threads to use for LFBGS during m-step. Default: 8\n\n");
//		label_numMstepThreads.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_numMstepThreads);
//		label_numEmissionCacheThreads = new JLabel("Number of emission cache threads ");
//		label_numEmissionCacheThreads.setToolTipText("Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.) Default: 8");
//		label_numEmissionCacheThreads.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_numEmissionCacheThreads);
//		label_numDecodeThreads = new JLabel("Number of decode threads ");
//		label_numDecodeThreads.setToolTipText("Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.) Default: 8");
//		label_numDecodeThreads.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_numDecodeThreads);
//		label_decodeBatchSize = new JLabel("Decode batch size ");
//		label_decodeBatchSize.setToolTipText("Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.) Default: 32");
//		label_decodeBatchSize.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_decodeBatchSize);
//		label_paddingMinWidth = new JLabel("Padding minimum width ");
//		label_paddingMinWidth.setToolTipText("Min horizontal padding between characters in pixels. (Best left at default value: 1.) Default: 1");
//		label_paddingMinWidth.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_paddingMinWidth);
//		label_paddingMaxWidth = new JLabel("Padding maximum width ");
//		label_paddingMaxWidth.setToolTipText("Max horizontal padding between characters in pixels (Best left at default value: 5.) Default: 5");
//		label_paddingMaxWidth.setHorizontalAlignment(SwingConstants.TRAILING);
//		panel_labels.add(label_paddingMaxWidth);
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
//		// names.foreach(n =>
//		// println(f"""input_$n%-30s = new JTextField(); panel_2.add(input_$n%-30s); input_$n%-30s.setColumns(10);"""))
//		input_inputDocPath = new JTextField();
//		input_inputDocPath.setToolTipText("Path of the directory that contains the input document images or pdfs. The entire directory will be recursively searched for any files that do not end in .txt (and that do not start with .). Required.");
//		panel_inputs.add(input_inputDocPath);
//		input_numDocs = new JTextField();
//		input_numDocs.setToolTipText("Number of documents to use. Ignore or use -1 to use all documents. Default: -1");
//		input_numDocs.setText("-1");
//		panel_inputs.add(input_numDocs);
//		input_inputLmPath = new JTextField();
//		input_inputLmPath.setToolTipText("Path to the language model file. Required.");
//		panel_inputs.add(input_inputLmPath);
//		input_inputFontPath = new JTextField();
//		input_inputFontPath.setToolTipText("Path of the font initializer file. Required.");
//		panel_inputs.add(input_inputFontPath);
//		input_trainFont = new JCheckBox();
//		input_trainFont.setToolTipText("Whether to learn the font from the input documents and write the font to a file. Default: false");
//		panel_inputs.add(input_trainFont);
//		input_trainFont.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent e) {
//				handletrainFontAction();
//			}
//		});
//		input_numEMIters = new JTextField();
//		input_numEMIters.setToolTipText("Number of iterations of EM to use for font learning. Default: 3");
//		input_numEMIters.setText("3");
//		panel_inputs.add(input_numEMIters);
//		input_outputPath = new JTextField();
//		input_outputPath.setToolTipText("Path of the directory that will contain output transcriptions and line extractions. Required.");
//		panel_inputs.add(input_outputPath);
//		input_extractedLinesPath = new JTextField();
//		input_extractedLinesPath.setToolTipText("Path of the directory where the line-extraction images should be read/written. If the line files exist here, they will be used; if not, they will be extracted and then written here. Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document. If ignored, the document will simply be read from the original document image file, and no line images will be written.\nDefault: null (Don't read or write line image files.)");
//		panel_inputs.add(input_extractedLinesPath);
//		input_outputFontPath = new JTextField();
//		input_outputFontPath.setToolTipText("Path to write the learned font file to. (Only if trainFont is set to true.) Required if trainFont=true, otherwise ignored.");
//		panel_inputs.add(input_outputFontPath);
//		input_outputLmPath = new JTextField();
//		input_outputLmPath.setToolTipText("Path to write the learned language model file to. (Only if trainFont is set to true.) Default: null (Don't write out the trained LM.)");
//		panel_inputs.add(input_outputLmPath);
//		input_allowLanguageSwitchOnPunct = new JCheckBox();
//		input_allowLanguageSwitchOnPunct.setToolTipText("A language model to be used to assign diacritics to the transcription output. Default: true");
//		panel_inputs.add(input_allowLanguageSwitchOnPunct);
//		input_binarizeThreshold = new JTextField();
//		input_binarizeThreshold.setText("0.12");
//		input_binarizeThreshold.setToolTipText(" Quantile to use for pixel value thresholding. (High values mean more black pixels.) Default: 0.12");
//		panel_inputs.add(input_binarizeThreshold);
//		input_crop = new JCheckBox();
//		input_crop.setSelected(true);
//		input_crop.setToolTipText("Crop pages? Useful when the image has a border around the page.");
//		panel_inputs.add(input_crop);
//		input_uniformLineHeight = new JCheckBox();
//		input_uniformLineHeight.setToolTipText("Scale all lines to have the same height? Default: true");
//		input_uniformLineHeight.setSelected(true);
//		panel_inputs.add(input_uniformLineHeight);
//		input_markovVerticalOffset = new JCheckBox();
//		input_markovVerticalOffset.setToolTipText("Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.) Default: false");
//		panel_inputs.add(input_markovVerticalOffset);
//		input_beamSize = new JTextField();
//		input_beamSize.setToolTipText("Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.) Default: 10");
//		input_beamSize.setText("10");
//		panel_inputs.add(input_beamSize);
//		input_emissionEngine = new ButtonGroup();
//		input_emissionEngine_default = new JRadioButton();
//		input_emissionEngine_default.setToolTipText("DEFAULT: Uses Java on CPU, which works on any machine but is the slowest method. ");
//		input_emissionEngine_default.setSelected(true);
//		input_emissionEngine_default.setText("Default");
//		panel_inputs.add(input_emissionEngine_default);
//		input_emissionEngine.add(input_emissionEngine_default);
//		input_emissionEngine_opencl = new JRadioButton();
//		input_emissionEngine_opencl.setToolTipText("OPENCL: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation..");
//		input_emissionEngine_opencl.setText("OpenCL");
//		panel_inputs.add(input_emissionEngine_opencl);
//		input_emissionEngine.add(input_emissionEngine_opencl);
//		input_emissionEngine_cuda = new JRadioButton();
//		input_emissionEngine_cuda.setToolTipText("CUDA: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.");
//		input_emissionEngine_cuda.setText("CUDA");
//		panel_inputs.add(input_emissionEngine_cuda);
//		input_emissionEngine.add(input_emissionEngine_cuda);
//		input_cudaDeviceID = new JTextField();
//		input_cudaDeviceID.setToolTipText("GPU ID when using CUDA emission engine. Default: 0\n\n");
//		input_cudaDeviceID.setText("0");
//		panel_inputs.add(input_cudaDeviceID);
//		input_numMstepThreads = new JTextField();
//		input_numMstepThreads.setToolTipText("Number of threads to use for LFBGS during m-step. Default: 8\n\n");
//		input_numMstepThreads.setText("8");
//		panel_inputs.add(input_numMstepThreads);
//		input_numEmissionCacheThreads = new JTextField();
//		input_numEmissionCacheThreads.setToolTipText("Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.) Default: 8");
//		input_numEmissionCacheThreads.setText("8");
//		panel_inputs.add(input_numEmissionCacheThreads);
//		input_numDecodeThreads = new JTextField();
//		input_numDecodeThreads.setToolTipText("Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.) Default: 8");
//		input_numDecodeThreads.setText("8");
//		panel_inputs.add(input_numDecodeThreads);
//		input_decodeBatchSize = new JTextField();
//		input_decodeBatchSize.setToolTipText("Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.) Default: 32");
//		input_decodeBatchSize.setText("32");
//		panel_inputs.add(input_decodeBatchSize);
//		input_paddingMinWidth = new JTextField();
//		input_paddingMinWidth.setToolTipText("Min horizontal padding between characters in pixels. (Best left at default value: 1.) Default: 1");
//		input_paddingMinWidth.setText("1");
//		panel_inputs.add(input_paddingMinWidth);
//		input_paddingMaxWidth = new JTextField();
//		input_paddingMaxWidth.setToolTipText("Max horizontal padding between characters in pixels (Best left at default value: 5.) Default: 5");
//		input_paddingMaxWidth.setText("5");
//		panel_inputs.add(input_paddingMaxWidth);
//		input_go = new JButton();
//		input_go.setMaximumSize(new Dimension(20, 100));
//		input_go.addActionListener(new ActionListener() {
//			@SuppressWarnings("static-access")
//			public void actionPerformed(ActionEvent e) {
//				TrainFont exec = new TrainFont();
//				// names.foreach(n =>
//				// println(f"""exec.$n%-30s = input_$n%-30s.getText();"""))
//				exec.inputDocPath = input_inputDocPath.getText();
//				exec.numDocs = Integer.valueOf(input_numDocs.getText());
//				exec.inputLmPath = input_inputLmPath.getText();
//				exec.inputFontPath = input_inputFontPath.getText();
//				exec.trainFont = input_trainFont.isSelected();
//				exec.numEMIters = Integer.valueOf(input_numEMIters.getText());
//				exec.outputPath = input_outputPath.getText();
//				exec.extractedLinesPath = input_extractedLinesPath.getText();
//				exec.outputFontPath = input_outputFontPath.getText();
//				exec.outputLmPath = input_outputLmPath.getText();
//				exec.allowLanguageSwitchOnPunct = input_allowLanguageSwitchOnPunct.isSelected();
//				exec.binarizeThreshold = Double.valueOf(input_binarizeThreshold.getText());
//				exec.crop = input_crop.isSelected();
//				exec.uniformLineHeight = input_uniformLineHeight.isSelected();
//				exec.markovVerticalOffset = input_markovVerticalOffset.isSelected();
//				exec.beamSize = Integer.valueOf(input_beamSize.getText());
//
//				EmissionCacheInnerLoopType input_emissionEngine_selected;
//				if (input_emissionEngine_default.isSelected())
//					input_emissionEngine_selected = EmissionCacheInnerLoopType.DEFAULT;
//				else if (input_emissionEngine_opencl.isSelected())
//					input_emissionEngine_selected = EmissionCacheInnerLoopType.OPENCL;
//				else if (input_emissionEngine_cuda.isSelected())
//					input_emissionEngine_selected = EmissionCacheInnerLoopType.CUDA;
//				else
//					throw new RuntimeException("No EmissionEngine selected");
//				// exec.emissionEngine =
//				// EmissionCacheInnerLoopType.valueOf(((JRadioButton)
//				// input_emissionEngine.getSelection().getSelectedObjects()[0]).getText());
//				exec.emissionEngine = input_emissionEngine_selected;
//
//				exec.cudaDeviceID = Integer.valueOf(input_cudaDeviceID.getText());
//				exec.numMstepThreads = Integer.valueOf(input_numMstepThreads.getText());
//				exec.numEmissionCacheThreads = Integer.valueOf(input_numEmissionCacheThreads.getText());
//				exec.numDecodeThreads = Integer.valueOf(input_numDecodeThreads.getText());
//				exec.decodeBatchSize = Integer.valueOf(input_decodeBatchSize.getText());
//				exec.paddingMinWidth = Integer.valueOf(input_paddingMinWidth.getText());
//				exec.paddingMaxWidth = Integer.valueOf(input_paddingMaxWidth.getText());
//				exec.run();
//			}
//		});
//		input_go.setText("Transcribe or Train Font");
//		panel_inputs.add(input_go);
//
//		handletrainFontAction();
//	}
//
//	private void handletrainFontAction() {
//		boolean active = input_trainFont.isSelected();
//
//		setEnabled(input_outputFontPath, active);
//		setEnabled(input_outputLmPath, active);
//
//		input_go.setText(input_trainFont.isSelected() ? "Train font" : "Transcribe");
//	}
//
//	public static void setEnabled(JTextField x, boolean enabled) {
//		Color fg = enabled ? Color.BLACK : Color.GRAY;
//		Color bg = enabled ? Color.WHITE : GRAY;
//		x.setEditable(enabled);
//		x.setForeground(fg);
//		x.setBackground(bg);
//	}

}
