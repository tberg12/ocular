package edu.berkeley.cs.nlp.ocular.image;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import indexer.Indexer;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.main.Main;

public class FontRenderer {
	
	public static Set<String> OUTLAWED_FONTS;
	static {
		OUTLAWED_FONTS = new HashSet<String>();
		OUTLAWED_FONTS.add("Vemana2000");
		OUTLAWED_FONTS.add("utkal");
		OUTLAWED_FONTS.add("Untitled1");
		OUTLAWED_FONTS.add("Symbol");
		OUTLAWED_FONTS.add("Standard Symbols L");
		OUTLAWED_FONTS.add("Saab");
		OUTLAWED_FONTS.add("Pothana2000");
		OUTLAWED_FONTS.add("OpenSymbol");
		OUTLAWED_FONTS.add("mry_KacstQurn");
		OUTLAWED_FONTS.add("Mallige");
		OUTLAWED_FONTS.add("Lohit Tamil");
		OUTLAWED_FONTS.add("Lohit Punjabi");
		OUTLAWED_FONTS.add("Lohit Hindi");
		OUTLAWED_FONTS.add("Lohit Gujarati");
		OUTLAWED_FONTS.add("Lohit Bengali");
		OUTLAWED_FONTS.add("Kedage");
		OUTLAWED_FONTS.add("KacstTitleL");
		OUTLAWED_FONTS.add("KacstTitle");
		OUTLAWED_FONTS.add("KacstScreen");
		OUTLAWED_FONTS.add("KacstQurn");
		OUTLAWED_FONTS.add("KacstPoster");
		OUTLAWED_FONTS.add("KacstPen");
		OUTLAWED_FONTS.add("KacstOne");
		OUTLAWED_FONTS.add("KacstOffice");
		OUTLAWED_FONTS.add("KacstNaskh");
		OUTLAWED_FONTS.add("KacstLetter");
		OUTLAWED_FONTS.add("KacstFarsi");
		OUTLAWED_FONTS.add("KacstDigital");
		OUTLAWED_FONTS.add("KacstDecorative");
		OUTLAWED_FONTS.add("KacstBook");
		OUTLAWED_FONTS.add("KacstArt");
		OUTLAWED_FONTS.add("GFS Solomos");
		OUTLAWED_FONTS.add("GFS Porson");
		OUTLAWED_FONTS.add("GFS Olga");
		OUTLAWED_FONTS.add("GFS Gazis");
		OUTLAWED_FONTS.add("GFS Didot Classic");
		OUTLAWED_FONTS.add("GFS Baskerville");
		OUTLAWED_FONTS.add("Dingbats");
		OUTLAWED_FONTS.add("GFS BodoniClassic");
		OUTLAWED_FONTS.add("Webdings");
		OUTLAWED_FONTS.add("Te X Gyre Chorus");
		OUTLAWED_FONTS.add("URW Chancery L");
		OUTLAWED_FONTS.add("TeXGyreChorus");
		OUTLAWED_FONTS.add("Lohit Devanagari");
		OUTLAWED_FONTS.add("Droid Sans Thai");
		OUTLAWED_FONTS.add("Droid Sans Hebrew");
		OUTLAWED_FONTS.add("Droid Sans Georgian");
		OUTLAWED_FONTS.add("Droid Sans Ethiopic");
		OUTLAWED_FONTS.add("Droid Sans Armenian");
		OUTLAWED_FONTS.add("Droid Arabic Naskh");
		OUTLAWED_FONTS.add("STIXIntegralsD");
		OUTLAWED_FONTS.add("STIXIntegralsSm");
		OUTLAWED_FONTS.add("STIXIntegralsUp");
		OUTLAWED_FONTS.add("STIXIntegralsUpD");
		OUTLAWED_FONTS.add("STIXIntegralsUpSm");
		OUTLAWED_FONTS.add("STIXNonUnicode");
		OUTLAWED_FONTS.add("STIXSizeFiveSym");
		OUTLAWED_FONTS.add("STIXSizeFourSym");
		OUTLAWED_FONTS.add("STIXSizeOneSym");
		OUTLAWED_FONTS.add("STIXSizeThreeSym");
		OUTLAWED_FONTS.add("STIXSizeTwoSym");
		OUTLAWED_FONTS.add("STIXVariants");
		OUTLAWED_FONTS.add("TakaoPGothic");
		OUTLAWED_FONTS.add("Droid Sans Japanese");
		OUTLAWED_FONTS.add("LKLUG");
		OUTLAWED_FONTS.add("Tibetan Machine Uni");
		OUTLAWED_FONTS.add("esint10");
		OUTLAWED_FONTS.add("eufm10");
		OUTLAWED_FONTS.add("cmex10");
		OUTLAWED_FONTS.add("cmsy10");
		OUTLAWED_FONTS.add("cmr10");
		OUTLAWED_FONTS.add("Rachana");
		OUTLAWED_FONTS.add("rsfs10");
		OUTLAWED_FONTS.add("wasy10");
	}

	public static void main(String[] args) {
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
	    for (String fontName : ge.getAvailableFontFamilyNames()) {
	    	if (!OUTLAWED_FONTS.contains(fontName)) {
	    		System.out.println(fontName);
	    		PixelType[][] data = renderString(fontName, "W", 30, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
	    		StringBuffer buf = new StringBuffer();
	    		if (data.length > 0) {
	    			for (int j=0; j<data[0].length; ++j) {
	    				for (int i=0; i<data.length; ++i) {
	    					PixelType val = data[i][j];
	    					if (val == PixelType.WHITE) {
	    						buf.append(". ");
	    					} else if (val == PixelType.BLACK) {
	    						buf.append("O ");
	    					} else if (val == PixelType.OBSCURED) {
	    						buf.append("X ");
	    					}
	    				}
	    				buf.append("\n");
	    			}
	    		}
	    		System.out.println(buf.toString());
	    	}
	    }
	}
	
	public static PixelType[][][][] getRenderedFont(Indexer<String> charIndexer, int height) {
		List<String> allowedFonts = new ArrayList<String>();
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
		for (String fontName : ge.getAvailableFontFamilyNames()) {
			if (!OUTLAWED_FONTS.contains(fontName)) {
				allowedFonts.add(fontName);
			}
		}
		
		StringBuffer alphabetStr = new StringBuffer();
		for (int c=0; c<charIndexer.size(); ++c) {
			alphabetStr.append(charIndexer.getObject(c));
		}
		
		PixelType[][][][] result = new PixelType[charIndexer.size()][][][];
		for (int c=0; c<charIndexer.size(); ++c) {
			List<PixelType[][]> rendered = new ArrayList<ImageUtils.PixelType[][]>();
			for (int f=0; f<allowedFonts.size(); ++f) {
				PixelType[][] renderedChar = renderString(allowedFonts.get(f), charIndexer.getObject(c), height, alphabetStr.toString());
				if (renderedChar.length > 0) rendered.add(renderedChar);
				else {
					if (!Charset.SPACE.equals(charIndexer.getObject(c))) { 
						System.out.println("Ignoring empty character rendering: "+allowedFonts.get(f)+", "+charIndexer.getObject(c));
					}
				}
			}
			result[c] = rendered.toArray(new PixelType[0][][]);
		}
		return result;
	}
	
	private static PixelType[][] renderString(String fontName, String s, int height, String alphabetString) {
		BufferedImage image = new BufferedImage(height, height, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = image.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setPaint(Color.WHITE);
		g.fillRect(0, 0, height, height);
		
		g.setPaint(Color.BLACK);
		Font font = new Font(fontName, Font.PLAIN, 10);
		g.setFont(font);
		FontMetrics fm = g.getFontMetrics(font);
		fm.getAscent();
		Rectangle2D strBounds = fm.getStringBounds(s, g);
		Rectangle2D alphabetBounds = fm.getStringBounds(alphabetString, g);
		AffineTransform affine = new AffineTransform();
		affine.translate(((double) height)/2.0 - strBounds.getCenterX()*((double) height)/alphabetBounds.getHeight(), ((double) height)/2.0 - alphabetBounds.getCenterY()*((double) height)/alphabetBounds.getHeight());
		affine.scale(((double) height)/alphabetBounds.getHeight(), ((double) height)/(alphabetBounds.getHeight()));
		g.setTransform(affine);
		g.drawString(s, 0, 0);

    	double[][] levels = ImageUtils.getLevels(image);
    	levels = horizontalCrop(levels);
    	return ImageUtils.getPixelTypes(levels);
	}
	
	private static double[][] horizontalCrop(double[][] levels) {
		double whitenessThresh = 0.4 * ImageUtils.MAX_LEVEL;
		int left = -1;
		for (int i=0; i<levels.length; ++i) {
			if (!allAbove(levels[i], whitenessThresh)) {
				left = i;
				break;
			}
		}
		int right = -1;
		for (int i=levels.length-1; i>=0; --i) {
			if (!allAbove(levels[i], whitenessThresh)) {
				right = i+1;
				break;
			}
		}
		if (left == -1) return new double[0][];
		double[][] result = new double[right-left][];
		for (int i=0; i<right-left; ++i) {
			result[i] = levels[left+i];
		}
		return result;
	}
	
	private static boolean allAbove(double[] vect, double threshold) {
		for (double val : vect) {
			if (val < threshold) {
				return false;
			}
		}
		return true;
	}
	
}
