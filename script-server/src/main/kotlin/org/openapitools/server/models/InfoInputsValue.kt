/**
* BON in a Box - Script service
* No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
*
* The version of the OpenAPI document: 1.0.0
* Contact: jean-michel.lord@mcgill.ca
*
* NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
* https://openapi-generator.tech
* Do not edit the class manually.
*/
package org.openapitools.server.models

/**
 * 
 * @param description 
 * @param label 
 * @param type 
 * @param example 
 */
data class InfoInputsValue(
    val description: kotlin.String? = null,
    val label: kotlin.String? = null,
    val type: kotlin.String? = null,
    // Any, since oneOf is not correctly supported by generator
    val example: Any? = null
) 

