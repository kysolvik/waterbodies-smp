"""Helper module defining labeling ontology for labelbox project

Not meant to be run directly
"""
import labelbox as lb

ontology_builder = lb.OntologyBuilder(
        tools=[
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="Natural"),
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="Reservatorios"),
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="Hidrelectrica"),
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="Mineracao")
            ],
        classifications=[
            lb.Classification(
                lb.Classification.Type.RADIO,
                name="Contains water?",
                options=[
                    lb.Option(value=0, label="No"),
                    lb.Option(value=1, label="Yes")
                    ]
                )
            ]
        )
