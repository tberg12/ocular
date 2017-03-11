package edu.berkeley.cs.nlp.ocular.data;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.SPACE;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.image.Visualizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Binarizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Cropper;
import edu.berkeley.cs.nlp.ocular.preprocessing.LineExtractor;
import edu.berkeley.cs.nlp.ocular.preprocessing.Straightener;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.last;
import tberg.murphy.fileio.f;

/**
 * A document that reads a file only as it is needed (and then stores
 * the contents in memory for later use).
 * 
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public abstract class LazyRawImageDocument implements Document {
	private final String inputPath;
	private final int lineHeight;
	private final double binarizeThreshold;
	private final boolean crop;

	private PixelType[][][] observations = null;

	private String extractedLinesPath = null;

	private String[][] diplomaticTextLines = null;
	private boolean diplomaticTextLinesLoaded = false;
	private String[][] normalizedTextLines = null;
	private boolean normalizedTextLinesLoaded = false;
	private List<String> normalizedText = null;
	private boolean normalizedTextLoaded = false;

	public LazyRawImageDocument(String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String extractedLinesPath) {
		this.inputPath = inputPath;
		this.lineHeight = lineHeight;
		this.binarizeThreshold = binarizeThreshold;
		this.crop = crop;
		this.extractedLinesPath = extractedLinesPath;
	}

	final public PixelType[][][] loadLineImages() {
	  if (observations == null) { // file has already been loaded in this Ocular run
		    if (extractedLinesPath == null) { // no pre-extraction path given
		    	observations = doLoadObservationsFromFile(); // load data from original file
		    }
		    else { // a pre-extraction path was given
		      if (extractionFilesPresent()) { // pre-extracted lines exist at the specified location
		      	observations = doLoadObservationsFromLineExtractionFiles(); // load data from pre-extracted line files
		      }
		      else { // pre-extraction has not been done yet; do it now.
		      	observations = doLoadObservationsFromFile(); // load data from original file
		      	writeExtractedLineImagesAggregateFile();
        		writeIndividualExtractedLineImageFiles(); // write extracted lines to files so they don't have to be re-extracted next time
		      }
	      }
    }
	  return observations;
	}

	private PixelType[][][] doLoadObservationsFromFile() {
		BufferedImage bi = doLoadBufferedImage();
		double[][] levels = ImageUtils.getLevels(bi);
		double[][] rotLevels = Straightener.straighten(levels);
		double[][] cropLevels = crop ? Cropper.crop(rotLevels, binarizeThreshold) : rotLevels;
		Binarizer.binarizeGlobal(binarizeThreshold, cropLevels);
		List<double[][]> lines = LineExtractor.extractLines(cropLevels);
		PixelType[][][] loadedObservations = new PixelType[lines.size()][][];
		for (int i = 0; i < lines.size(); ++i) {
			loadedObservations[i] = imageToObservation(ImageUtils.makeImage(lines.get(i)));
		}
		return loadedObservations;
	}
	
	private PixelType[][][] doLoadObservationsFromLineExtractionFiles() {
		System.out.println("Loading pre-extracted line images from " + leLineDir());
		final Pattern pattern = Pattern.compile("line(\\d+)\\." + ext());
		File[] lineImageFiles = new File(leLineDir()).listFiles(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return pattern.matcher(name).matches();
			}
		});
		if (lineImageFiles == null) throw new RuntimeException("lineImageFiles is null");
		if (lineImageFiles.length == 0) throw new RuntimeException("lineImageFiles.length == 0");
		Arrays.sort(lineImageFiles);

		PixelType[][][] loadedObservations = new PixelType[lineImageFiles.length][][];
		for (int i = 0; i < lineImageFiles.length; ++i) {
			Matcher m = pattern.matcher(lineImageFiles[i].getName());
			if (m.find() && Integer.valueOf(m.group(1)) != i) throw new RuntimeException("Trying to load lines from "+leLineDir()+" but the file for line "+i+" is missing (found "+m.group(1)+" instead).");
			String lineImageFile = fullLeLinePath(i);
			System.out.println("    Loading pre-extracted line from " + lineImageFile);
			try {
				loadedObservations[i] = imageToObservation(f.readImage(lineImageFile));
			}
			catch (Exception e) {
				throw new RuntimeException("Couldn't read line image from: " + lineImageFile, e);
			}
		}
		return loadedObservations;
	}
	
	private PixelType[][] imageToObservation(BufferedImage image) {
		if (lineHeight >= 0) {
			return ImageUtils.getPixelTypes(ImageUtils.resampleImage(image, lineHeight));
		}
		else {
			return ImageUtils.getPixelTypes(image);
		}
	}

	/**
	 * Write all extracted lines to a single file for easy viewing
	 * 
	 * @multilineExtractionImagePath The path of the file to write to.
	 */
	public void writeExtractedLineImagesAggregateFile(String multilineExtractionImagePath) {
		System.out.println("Writing file line-extraction image to: " + multilineExtractionImagePath);
		new File(multilineExtractionImagePath).getAbsoluteFile().getParentFile().mkdirs();
		f.writeImage(multilineExtractionImagePath, Visualizer.renderLineExtraction(observations));
	}
	
	/**
	 * Write all extracted lines to a single file for easy viewing
	 */
	public void writeExtractedLineImagesAggregateFile() {
		writeExtractedLineImagesAggregateFile(multilineExtractionImagePath());
	}
	
	public void writeIndividualExtractedLineImageFiles() {
		new File(leLineDir()).mkdirs();
		for (int l = 0; l < observations.length; ++l) {
			PixelType[][] observationLine = observations[l];
			String linePath = fullLeLinePath(l);
			System.out.println("  Writing individual line-extraction image to: " + linePath);
			f.writeImage(linePath, Visualizer.renderLineExtraction(observationLine));
		}
	}
	
	private boolean extractionFilesPresent() {
		File f = new File(fullLeLinePath(0));
		System.out.println("Looking for extractions in ["+f+"]. "+(f.exists() ? "Found" : "Not found")+".");
		return f.exists();
	}
	
	private String[][] loadTextFile(File textFile, String name) {
		if (textFile.exists()) {
			System.out.println("Evaluation "+name+" text found at " + textFile);
			List<List<String>> textList = new ArrayList<List<String>>();
			try {
				BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(textFile), "UTF-8"));
				while (in.ready()) {
					textList.add(Charset.readNormalizeCharacters(in.readLine()));
				}
				in.close();
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}

			String[][] textLines = new String[textList.size()][];
			for (int i = 0; i < textLines.length; ++i) {
				List<String> line = textList.get(i);
				textLines[i] = line.toArray(new String[line.size()]);
			}
			return textLines;
		}
		else {
			System.out.println("No evaluation "+name+" text found at " + textFile + "  (This is only a problem if you were trying to provide a gold "+name+" transcription to check accuracy.)");
			return null;
		}
	}
	
	public String[][] loadDiplomaticTextLines() {
		if (!diplomaticTextLinesLoaded) {
			diplomaticTextLines = loadTextFile(new File(baseName().replaceAll("\\.[^.]*$", "") + ".txt"), "diplomatic");
		}
		diplomaticTextLinesLoaded = true;
		return diplomaticTextLines;
	}

	public String[][] loadNormalizedTextLines() {
		if (!normalizedTextLinesLoaded) {
			normalizedTextLines = loadTextFile(new File(baseName().replaceAll("\\.[^.]*$", "") + "_normalized.txt"), "normalized");
		}
		normalizedTextLinesLoaded = true;
		return normalizedTextLines;
	}

	public List<String> loadNormalizedText() {
		if (!normalizedTextLoaded) {
			String[][] normalizedTextLines = loadNormalizedTextLines();
			if (normalizedTextLines != null) {
				normalizedText = new ArrayList<String>();
				for (String[] lineChars : loadNormalizedTextLines()) {
					for (String c : lineChars) {
						if (SPACE.equals(c) && (normalizedText.isEmpty() || SPACE.equals(last(normalizedText)))) {
							// do nothing -- collapse spaces
						}
						else {
							normalizedText.add(c);
						}
					}
					if (!normalizedText.isEmpty() && !SPACE.equals(last(normalizedText))) {
						normalizedText.add(SPACE);
					}
				}
				if (SPACE.equals(last(normalizedText))) {
					normalizedText.remove(normalizedText.size()-1);
				}
			}
		}
		normalizedTextLoaded = true;
		return normalizedText;
	}

	private String multilineExtractionImagePath() { return fullLePreExt() + "." + ext(); }
	private String leLineDir() { return fullLePreExt() + "_" + ext(); }
	private String fileParent() { return FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), file())._2; }
	private String fullLePreExt() { return extractedLinesPath + "/" + fileParent() + "/" + preext() + "-line_extract"; }
	private String fullLeLinePath(int lineNum) { return String.format(leLineDir() + "/line%02d." + ext(), lineNum); }
	
	abstract protected File file();
	abstract protected BufferedImage doLoadBufferedImage();
	abstract protected String preext();
	abstract protected String ext();
	
}
