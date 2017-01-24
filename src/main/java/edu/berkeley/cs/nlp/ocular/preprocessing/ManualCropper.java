package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.MouseInputAdapter;

import tberg.murphy.fileio.f;
  
public class ManualCropper extends JPanel {
	private static final long serialVersionUID = 1L;
	
	BufferedImage image;
    Dimension size;
    Rectangle clip;
    int curSeg;
    int curImage;
    String[] names;
    String path;
  
    public ManualCropper(String path, String[] names)
    {
    	this.path = path;
    	this.names = names;
    	curImage = -1;
    	loadNextImage();
    }
    
    private void loadPrevImage() {
    	curImage = Math.max(0, curImage-1);
    	
        File file = new File(path + "/" + names[curImage]);
        try {
			this.image = ImageIO.read(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
        
    	this.curSeg = 0;
    	if (curImage == 0) {
    		size = new Dimension(image.getWidth(), image.getHeight());
    		createClip();
    		setClipTopLeft(0,0);
    	}
    	
    	repaint();
    }
    
    private void loadNextImage() {
    	curImage = Math.min(names.length-1, curImage+1);
    	
        File file = new File(path + "/" + names[curImage]);
        try {
			this.image = ImageIO.read(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
        
    	this.curSeg = 0;
    	if (curImage == 0) {
    		size = new Dimension(image.getWidth(), image.getHeight());
    		createClip();
    		setClipTopLeft(0,0);
    	}
    	
    	repaint();
    }
  
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);
        int x = (getWidth() - size.width)/2;
        int y = (getHeight() - size.height)/2;
        g2.drawImage(image, x, y, this);
        if(clip == null)
        	createClip();
        g2.setPaint(Color.red);
        g2.draw(clip);
    }
    
    public void setClipBottomRight(int w, int h) {
        // keep clip within raster
        int x0 = (getWidth() - size.width)/2;
        int y0 = (getHeight() - size.height)/2;
        int x = clip.x;
        int y = clip.y;
        if(x < x0 || x + w  > x0 + size.width ||
           y < y0 || y + h > y0 + size.height)
            return;
        clip = new Rectangle(w, h);
        clip.x = x;
        clip.y = y;
        clip.setLocation(x, y);
        repaint();
    }
  
    public void setClipTopLeft(int x, int y)
    {
        // keep clip within raster
        int x0 = (getWidth() - size.width)/2;
        int y0 = (getHeight() - size.height)/2;
        if(x < x0 || x + clip.width  > x0 + size.width ||
           y < y0 || y + clip.height > y0 + size.height)
            return;
        clip.setLocation(x, y);
        repaint();
    }
  
    public Dimension getPreferredSize()
    {
        return size;
    }
  
    private void createClip()
    {
        clip = new Rectangle(140, 140);
        clip.x = 0;
        clip.y = 0;
    }
  
    private void clipImage()
    {
        BufferedImage clipped = null;
        try
        {
            int w = clip.width;
            int h = clip.height;
            int x0 = (getWidth()  - size.width)/2;
            int y0 = (getHeight() - size.height)/2;
            int x = clip.x - x0;
            int y = clip.y - y0;
            clipped = image.getSubimage(x, y, w, h);
        }
        catch(RasterFormatException rfe)
        {
            System.out.println("raster format error: " + rfe.getMessage());
            return;
        }
        String baseName = (names[curImage].lastIndexOf('.') == -1) ? names[curImage] : names[curImage].substring(0, names[curImage].lastIndexOf('.'));
        f.writeImage(path + "/" + baseName + "_seg"+curSeg+".png", clipped);
//        JLabel label = new JLabel(new ImageIcon(clipped));
//        JOptionPane.showMessageDialog(this, label, "clipped image", JOptionPane.PLAIN_MESSAGE);
        curSeg++;
    }
  
    private JPanel getUIPanel()
    {
    	JPanel panel = new JPanel();
        JButton clip = new JButton("clip image");
        clip.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                clipImage();
            }
        });
        panel.add(clip);
        JButton prev = new JButton("prev image");
        prev.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                loadPrevImage();
            }
        });
        panel.add(prev);
        JButton next = new JButton("next image");
        next.addActionListener(new ActionListener()
        {
        	public void actionPerformed(ActionEvent e)
        	{
        		loadNextImage();
        	}
        });
        panel.add(next);
        return panel;
    }
  
    public static void main(String[] args) throws IOException {
		String path = args[0];
		File dir = new File(path);
		String[] names = dir.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return name.endsWith(".png") || name.endsWith(".jpg");
			}
		});
        ManualCropper test = new ManualCropper(path, names);
        ClipMover mover = new ClipMover(test);
        test.addMouseListener(mover);
        test.addMouseMotionListener(mover);
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.getContentPane().add(new JScrollPane(test));
        f.getContentPane().add(test.getUIPanel(), "South");
        f.setSize(2400,1200);
        f.setLocation(0,0);
        f.setVisible(true);
    }
}
  
class ClipMover extends MouseInputAdapter
{
    ManualCropper cropping;
    Point offset;
  
    public ClipMover(ManualCropper c)
    {
        cropping = c;
        offset = new Point();
    }
  
    public void mousePressed(MouseEvent e)
    {
        Point p = e.getPoint();
        if (e.isShiftDown()) {
        	cropping.setClipBottomRight(Math.max(10, p.x - cropping.clip.x), Math.max(10, p.y - cropping.clip.y));
        } else {
        	cropping.setClipTopLeft(p.x, p.y);
        }
    }
  
    public void mouseReleased(MouseEvent e)
    {
    }
  
    public void mouseDragged(MouseEvent e)
    {
    }
}