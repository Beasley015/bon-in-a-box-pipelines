package org.geobon.pipeline

class Output(override val type:String) : Pipe {

    var step: Step? = null
    var value:String? = null

    override fun pull(): String {
        if(value == null) {
            step?.apply { execute() }
                ?: throw Exception("Output disconnected from any step")
        }
        return value ?: throw Exception("Output has not been set by step")
    }
}