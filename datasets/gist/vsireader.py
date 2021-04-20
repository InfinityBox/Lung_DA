import bioformats
from bioformats import ImageReader
import bioformats.omexml as ome
import xml.dom.minidom as Minidom
import numpy as np


class VsiReader(bioformats.formatreader.ImageReader):
    def __init__(self, path, url=None, perform_init=True, layer_init: int=0,  tilesize_init=(1000, 1000)):
        super(VsiReader, self).__init__(path, url, perform_init)
        self.channels = self.rdr.getSizeC()
        self.layersCount = self.rdr.getSeriesCount()
        self.initial(layer=layer_init, tilesize=tilesize_init)
        self.metadata = bioformats.get_omexml_metadata(path=path)

    def __delete__(self):

        self.close()

    def initial(self, tilesize: tuple or list, layer: int=0):

        self.setLayer(layer)
        self.setTileSize(tileSize=tilesize)

    def setLayer(self, layerIndex:int):
        assert 0 <= layerIndex < self.layersCount, "ParamError: layer [{}] is out of the range".format(layerIndex)

        self.currentLayer = layerIndex
        self.rdr.setSeries(self.currentLayer)
        self.currentLayerWidth = self.rdr.getSizeX()
        self.currentLayerHeight = self.rdr.getSizeY()

    def setTileSize(self, tileSize: tuple or list):
        """
        (width, height)
        """
        assert len(tileSize) == 2, "ParamError: tilesize must be a tuple or list with 2 elements"

        self.widthTile = tileSize[0]
        self.heightTile = tileSize[1]
        self.numTileX = self.currentLayerWidth // self.widthTile + \
                        int(self.currentLayerWidth % self.widthTile > 0)
        self.numTileY = self.currentLayerHeight // self.heightTile + \
                        int(self.currentLayerHeight % self.heightTile > 0)

    def getMetadata(self):
        return self.metadata   # a str of xml

    def getCurrentLayer(self):

        return self.currentLayer

    def getCurrentSize(self):

        return self.currentLayerWidth, self.currentLayerHeight

    def getSize(self, layer: int=None):

        assert -self.layersCount < layer < self.layersCount, \
            "layer [{}] must in range ({},{})".format(layer, -self.layersCount, self.layersCount)

        if layer is not None:
            layer_origin = self.rdr.getSeries()
            layer = layer + self.layersCount - 1 if layer < 0 else layer
            # transfer negative layer index to positive layer index
            self.rdr.setSeries(layer)
            width = self.rdr.getSizeX()
            height = self.rdr.getSizeY()
            self.rdr.setSeries(layer_origin)
        else:
            width, height = self.getCurrentSize()

        return width, height

    def getTileSize(self):

        return self.widthTile, self.heightTile

    def getNumTile(self):

        return self.numTileX, self.numTileY

    def getTileField(self, indexTileX, indexTileY):

        assert (0 <= indexTileX < self.numTileX) and (0 <= indexTileY < self.numTileY), "index of tile is out of range"

        if indexTileX < self.numTileX - 1:
            if indexTileY < self.numTileY - 1:
                widthTile = self.widthTile
                heightTile = self.heightTile
            else:
                heightTile = self.currentLayerHeight - (self.currentLayerHeight // self.heightTile) * self.heightTile
                widthTile = self.widthTile
        else:
            if indexTileY < self.numTileY - 1:
                heightTile = self.heightTile
                widthTile = self.currentLayerWidth - (self.currentLayerWidth // self.widthTile) * self.widthTile
            else:
                heightTile = self.currentLayerHeight - (self.currentLayerHeight // self.heightTile) * self.heightTile
                widthTile = self.currentLayerWidth - (self.currentLayerWidth // self.widthTile) * self.widthTile

        return (indexTileX * self.widthTile, indexTileY * self.heightTile,
                indexTileX * self.widthTile + widthTile, indexTileY * self.heightTile + heightTile)

    def getTile(self, indexTileX, indexTileY):

        assert (0 <= indexTileX < self.numTileX) and (0 <= indexTileY < self.numTileY), "index of tile is out of range"

        if indexTileX < self.numTileX - 1:
            if indexTileY < self.numTileY - 1:
                widthTile = self.widthTile
                heightTile = self.heightTile
                bytes_image = self.rdr.openBytesXYWH(0, indexTileX * self.widthTile, indexTileY * self.heightTile,
                                                       widthTile, heightTile)
                image = bytes_image.reshape(heightTile, widthTile, self.channels)
            else:
                heightTile = self.currentLayerHeight - (self.currentLayerHeight // self.heightTile) * self.heightTile
                widthTile = self.widthTile
                bytes_image = self.rdr.openBytesXYWH(0, indexTileX * self.widthTile, indexTileY * self.heightTile,
                                                       widthTile, heightTile)
                image = bytes_image.reshape(heightTile, widthTile, self.channels)
        else:
            if indexTileY < self.numTileY - 1:
                heightTile = self.heightTile
                widthTile = self.currentLayerWidth - (self.currentLayerWidth // self.widthTile) * self.widthTile
                bytes_image = self.rdr.openBytesXYWH(0, indexTileX * self.widthTile, indexTileY * self.heightTile,
                                                       widthTile, heightTile)
                image = bytes_image.reshape(heightTile, widthTile, self.channels)
            else:
                heightTile = self.currentLayerHeight - (self.currentLayerHeight // self.heightTile) * self.heightTile
                widthTile = self.currentLayerWidth - (self.currentLayerWidth // self.widthTile) * self.widthTile
                bytes_image = self.rdr.openBytesXYWH(0, indexTileX * self.widthTile, indexTileY * self.heightTile,
                                                       widthTile, heightTile)
                image = bytes_image.reshape(heightTile, widthTile, self.channels)

        return image, ((indexTileX * self.widthTile, indexTileY * self.heightTile), (widthTile, heightTile))

    def getImage(self, layer: int=None):
        """
        get full image of specific layer, this function is supported by inverted order

        Parameters
        ----------
        layer : int
            the specific layer, it must be in the range [-self.layersCount, self.layersCount)

        Returns
        -------
        image : np.ndarray
            the full image of specific layer,
            whose dtype is uint8, range is [0, 255]
        """
        assert -self.layersCount < layer < self.layersCount, \
            "layer [{}] must in range ({},{})".format(layer, -self.layersCount, self.layersCount)

        sizeC, sizeZ, sizeT = 0, 0, 0
        layer_origin = self.rdr.getSeries()
        if layer is not None:
            layer = layer + self.layersCount - 1 if layer < 0 else layer
            # transfer negative layer index to positive layer index
            self.rdr.setSeries(layer)
            image = self.read(sizeC, sizeZ, sizeT, layer)
            self.rdr.setSeries(layer_origin)
        else:
            image = self.read(sizeC, sizeZ, sizeT, layer_origin)

        return np.array((image * 255).astype(np.uint8))

    def getRegion(self, xywh: tuple or list, layer: int=None):
        """
        get image region of specific layer, this function is supported by inverted order

        Parameters
        ----------
        xywh : tuple or list
            (x, y) of top-left point of image on the specific layer
            (w, h) is the width and height of image region
            x,y,w,h can't be float, but it will be converted to int by around
        layer : int
            the specific layer, it must be in the range [-self.layersCount, self.layersCount)

        Returns
        -------
        image : np.ndarray
            the full image of specific layer,
            whose dtype is uint8, range is [0, 255]
        """

        assert -self.layersCount < layer < self.layersCount, \
            "layer [{}] must in range ({},{})".format(layer, -self.layersCount, self.layersCount)

        xywh = tuple([int(np.round(ele)) for ele in xywh])
        sizeC, sizeZ, sizeT = 0, 0, 0
        layer_origin = self.rdr.getSeries()
        if layer is not None:
            layer = layer + self.layersCount - 1 if layer < 0 else layer
            # transfer negative layer index to positive layer index
            self.rdr.setSeries(layer)
            image = self.read(sizeC, sizeZ, sizeT, layer, XYWH=xywh)
            self.rdr.setSeries(layer_origin)
        else:
            image = self.read(sizeC, sizeZ, sizeT, layer_origin, XYWH=xywh)

        return np.array((image * 255).astype(np.uint8))


class XmlParser(object):
    def __init__(self, xml):
        assert isinstance(xml, str), "xml must be a str"
        self.strXml = xml

    def vsi_infoextract(self):
        domTree = Minidom.parseString(self.strXml)
        elementInstrument = domTree.getElementsByTagName("Instrument")
        elementsImage = domTree.getElementsByTagName("Image")

        listImagesinfo = list()
        for index, elementImage in enumerate(elementsImage):
            imageinfo = {}
            imageinfo.setdefault("Name", elementImage.getAttribute("Name"))
            nodelistDate = elementImage.getElementsByTagName("AcquisitionDate")
            nodelistBasic = elementImage.getElementsByTagName("Pixels")

            # Basic Info Part 1
            imageinfo.setdefault("SizeY", nodelistBasic[0].getAttribute("SizeY"))
            imageinfo.setdefault("SizeX", nodelistBasic[0].getAttribute("SizeX"))

            # Extral Info
            if nodelistDate.length == 0:
                imageinfo.setdefault("AcquisitionDate", "-")
                imageinfo.setdefault("PhysicalSizeY", "-")
                imageinfo.setdefault("PhysicalSizeX", "-")
            else:
                imageinfo.setdefault("AcquisitionDate", nodelistDate[0].firstChild.nodeValue)
                imageinfo.setdefault("PhysicalSizeY", nodelistBasic[0].getAttribute("PhysicalSizeY"))
                imageinfo.setdefault("PhysicalSizeX", nodelistBasic[0].getAttribute("PhysicalSizeX"))

            # Basic Info Part 2
            imageinfo.setdefault("DimensionOrder", nodelistBasic[0].getAttribute("DimensionOrder"))
            imageinfo.setdefault("Type", nodelistBasic[0].getAttribute("Type"))
            imageinfo.setdefault("SizeZ", nodelistBasic[0].getAttribute("SizeZ"))
            imageinfo.setdefault("SizeC", nodelistBasic[0].getAttribute("SizeC"))
            imageinfo.setdefault("SizeT", nodelistBasic[0].getAttribute("SizeT"))
            imageinfo.setdefault("BigEndian", nodelistBasic[0].getAttribute("BigEndian"))
            imageinfo.setdefault("SignificantBits", nodelistBasic[0].getAttribute("SignificantBits"))

            listImagesinfo.append(imageinfo)

        return listImagesinfo


if __name__ == "__main__":

    import pprint
    import matplotlib.pyplot as plt

    reader = VsiReader(path="I:\\GIST\\sorted\\025-394944-01.vsi")
    strXml = reader.getMetadata()
    parser = XmlParser(strXml)
    info = parser.vsi_infoextract()
    domTree = Minidom.parseString(strXml)
    collection = domTree.documentElement
    infoImages = domTree.getElementsByTagName("Image")
    for infoImage in infoImages:
        if infoImage.hasAttribute("Name"):
            print("Name: ", infoImage.getAttribute("Name"))
    print("Done.")
