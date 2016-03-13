package edu.berkeley.cs.nlp.ocular.data;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface Document {
	public String baseName();
	public PixelType[][][] loadLineImages();
	public String[][] loadDiplomaticTextLines();
	public String[][] loadNormalizedTextLines();
	public List<String> loadNormalizedText();
}
