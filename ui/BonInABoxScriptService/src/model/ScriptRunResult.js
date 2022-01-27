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
 *
 */

import ApiClient from '../ApiClient';

/**
 * The ScriptRunResult model module.
 * @module model/ScriptRunResult
 * @version 1.0.0
 */
class ScriptRunResult {
    /**
     * Constructs a new <code>ScriptRunResult</code>.
     * @alias module:model/ScriptRunResult
     */
    constructor() { 
        
        ScriptRunResult.initialize(this);
    }

    /**
     * Initializes the fields of this object.
     * This method is used by the constructors of any subclasses, in order to implement multiple inheritance (mix-ins).
     * Only for internal use.
     */
    static initialize(obj) { 
    }

    /**
     * Constructs a <code>ScriptRunResult</code> from a plain JavaScript object, optionally creating a new instance.
     * Copies all relevant properties from <code>data</code> to <code>obj</code> if supplied or a new instance if not.
     * @param {Object} data The plain JavaScript object bearing properties of interest.
     * @param {module:model/ScriptRunResult} obj Optional instance to populate.
     * @return {module:model/ScriptRunResult} The populated <code>ScriptRunResult</code> instance.
     */
    static constructFromObject(data, obj) {
        if (data) {
            obj = obj || new ScriptRunResult();

            if (data.hasOwnProperty('logs')) {
                obj['logs'] = ApiClient.convertToType(data['logs'], 'String');
            }
            if (data.hasOwnProperty('files')) {
                obj['files'] = ApiClient.convertToType(data['files'], {'String': 'String'});
            }
        }
        return obj;
    }


}

/**
 * @member {String} logs
 */
ScriptRunResult.prototype['logs'] = undefined;

/**
 * @member {Object.<String, String>} files
 */
ScriptRunResult.prototype['files'] = undefined;






export default ScriptRunResult;
