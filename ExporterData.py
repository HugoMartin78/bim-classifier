# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:49:07 2015

@author: Hugo
"""

import fbx
import numpy
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

#Place où se trouve le fichier FBX, mis en pièce jointe dans le mail
filepath = r'\FBXs\PJP.fbx'
nametxt = 'PJP.txt'

#Chargement global de la scene
manager = fbx.FbxManager.Create()
importer = fbx.FbxImporter.Create(manager, 'myImporter')
status = importer.Initialize(filepath)
scene = fbx.FbxScene.Create( manager, 'myScene' )
importer.Import( scene )
importer.Destroy()

#Toutes les données seront mises dans le tableaux export
Export = []

#Le Label est rempli par la suite dans le code Labeller, cependant il possède une place dans le code
Label = "K"

def findAmplitudeVolume(ControlPoints):
    #Les sommets sont gérés par des fbxvector qui ne sont pas des tableaux ou des listes, on doit donc les extraires et le comparer par boucle :-()
    #Cette foncion permet de définir l'échelle de taille de l'objet par ses sommets les plus éloignés.    
        
    result = []
    
    for i in range(0,3):
        Max = ControlPoints[0][i]
        Min = ControlPoints[0][i]
        for index in ControlPoints:
            if index[i] > Max:
                Max = index[i]
            if index[i] < Min:
                Min = index[i]
        result.append(Max-Min)

    resultat = result[0] * result[1] * result[2]
    
    return round(resultat,2)


def findObjectProperty( node, Export, currentPath = [] ):
    
    #Déclaration Variable pour element vide
    MaterialName = "X"
    MyTexture = "X" 
    PolyNumber = 0
    Size = 0
    
    #Pour le nom de l'objet (le nom de ses père se retrouve devant)
    currentPath.append( node.GetName() )

    #Retrouve le nombre de polygon de l'objet*
    Mesh = node.GetMesh()
    if (Mesh):
        PolyNumber =  Mesh.GetPolygonCount()
        
    #Caluler la dimension de l'enveloppe de l'objet
    Geometry = node.GetGeometry()
    if (Geometry):
        Size = findAmplitudeVolume(Geometry.GetControlPoints())
    
    
    #Pour le nom du materiau placé sur l'objet
    for materialIndex in range( 0, node.GetMaterialCount() ):
        material = node.GetMaterial( materialIndex )
        MaterialName = MaterialName + " " + material.GetName()
        
    #Pour retourver le nom de la texture
        for propertyIndex in range( 0, fbx.FbxLayerElement.sTypeTextureCount() ):
            #Les propriété du matériel ne nous interesse pas car cela n'est jamais significatif sur les modèles 3D
            property = material.FindProperty( fbx.FbxLayerElement.sTextureChannelNames( propertyIndex ) )
            
            for textureIndex in range( 0, property.GetSrcObjectCount( fbx.FbxFileTexture.ClassId ) ):                
                texture = property.GetSrcObject( fbx.FbxFileTexture.ClassId, textureIndex )
                MyTexture = texture.GetRelativeFileName()
                
                
                
    #Inscription dans le tableau uniquement lorsqu'il s'agit d'un objet
    if(PolyNumber != 0): 
        
        #Suppression des # qui posent problèmes dans la créatin de tableaux
        
        MaterialName = MaterialName.replace("#", "")
        Export.append ([node.GetName().encode('utf-8'), MaterialName, PolyNumber, Size, MyTexture.encode('utf-8'), Label])

    
    #La fonction est récrusive pour tous les fils de chaque element
    for i in range( 0, node.GetChildCount() ):
        findObjectProperty( node.GetChild( i ), Export, currentPath )
    
    #Un fois un père finis, on supprimer son nom
    currentPath.pop()
        
        
findObjectProperty( scene.GetRootNode(), Export )

numpy.savetxt(nametxt,Export, fmt = '%s', delimiter = '|')


